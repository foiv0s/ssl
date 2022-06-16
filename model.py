import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sinkhorn
from graphs import ResNet, BasicBlock, Bottleneck, MLPClassifier, Projection, Prototypes, \
    ProjectionCNN, Predictor, SpectralNorm
from torch import autograd
from matplotlib import pyplot as plt


class Encoder(nn.Module):
    def __init__(self, encoder_size=32, project_dim=128, model_type='resnet18', batch=True, h=1):
        super(Encoder, self).__init__()
        # encoding block for local features
        print('Using a {}x{} encoder'.format(encoder_size, encoder_size))
        inplanes = 64
        if encoder_size == 32:
            conv1 = nn.Sequential(nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(inplanes),
                                  nn.ReLU(inplace=True))
        elif encoder_size == 96 or encoder_size == 64:
            conv1 = nn.Sequential(nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(inplanes),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        elif encoder_size == 224:
            inplanes = 64
            conv1 = nn.Sequential(nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(inplanes),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(encoder_size))

        if model_type == 'resnet18':
            # ResNet18 block
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], conv1)
        elif model_type == 'resnet34':
            self.model = ResNet(BasicBlock, [3, 4, 6, 3], conv1)
        elif model_type == 'resnet50':
            self.model = ResNet(Bottleneck, [3, 4, 6, 3], conv1)
        else:
            raise RuntimeError("Wrong model type")

        print(self.get_param_n())

        dummy_batch = torch.zeros((2, 3, encoder_size, encoder_size))
        rkhs_1 = self.model(dummy_batch)[-1]
        self.emb_dim = rkhs_1.size(1)
        self.project = Projection(self.emb_dim, project_dim, self.emb_dim, batch=batch, h=h)

    def get_param_n(self):
        w = 0
        for p in self.model.parameters():
            w += np.product(p.shape)
        return w

    def forward(self, x):
        layers = self.model(x)
        z = torch.flatten(layers[-1], 1)
        return z, self.project(z)

    def forward_k(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, n_classes, encoder_size=32, prototypes=1000, project_dim=128, temp=0.1, eps=0.05, epoch=500,
                 mom=0.99, loss_type=0, model_type='resnet18', batch_mlp=True, hidden_n=1, mem_bank_n=10, logger=None):
        super(Model, self).__init__()

        self.hyperparams = {
            'n_classes': n_classes,
            'encoder_size': encoder_size,
            'prototypes': prototypes,
            'project_dim': project_dim,
            'temp': temp,
            'eps': eps,
            'mom': mom,
            'epoch': epoch,
            'loss_type': loss_type,
            'batch_mlp': batch_mlp,
            'hidden_n': hidden_n,
            'model_type': model_type,
            'mem_bank_n': mem_bank_n,
            'logger': logger
        }
        self.logger = logger

        self.encoder = Encoder(encoder_size=encoder_size, project_dim=project_dim, model_type=model_type,
                               batch=batch_mlp, h=hidden_n)
        self.encoder_k = Encoder(encoder_size=encoder_size, project_dim=project_dim, model_type=model_type,
                                 batch=batch_mlp, h=hidden_n)
        self.prototypes = Prototypes(project_dim, prototypes)
        self.predictor = Projection(project_dim, project_dim, h=hidden_n, batch=batch_mlp)

        self.evaluator = MLPClassifier(n_classes, self.encoder.emb_dim)
        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False

        self.mlp_modules = [self.encoder.project, self.predictor]
        self.cnn_module = [self.encoder.model]
        self.class_modules = [self.evaluator]
        self._t, self._e = temp, eps
        self.m, self.loss_type = mom, loss_type
        self.epoch = epoch
        self.mem_bank = None
        self.mem_bank_n = mem_bank_n

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param.data * (1 - self.m) + param_k.data * self.m

    def _encode_nce(self, res_dict, aug_imgs, num_crops):
        b = aug_imgs[0].size(0)
        res_dict['Z'], h = [], []

        for i, aug_imgs_ in enumerate(aug_imgs):
            z_, h_ = self.encoder(aug_imgs_)
            res_dict['Z'].append(z_), h.append(F.normalize(h_, 2, 1))

        res_dict['loss'] = []
        mat = torch.matmul(h[0], torch.cat(h).T) / self._t
        mask_pos = torch.eye(b, dtype=torch.bool, device='cuda').repeat(1, np.sum(num_crops))
        mask_neg = torch.logical_xor(torch.tensor([True]).cuda(), mask_pos)
        pos = mat.masked_select(mask_pos).view(b, -1)[:, 1:]
        neg = mat.masked_select(mask_neg).view(b, -1).exp().sum(-1, keepdims=True)
        res_dict['loss'] = -(pos - torch.log(neg + pos.exp())).mean()
        res_dict['Z'] = torch.cat(res_dict['Z'])
        res_dict['class'] = self.evaluator(res_dict['Z'][:b * num_crops[0]])

        return res_dict

    def _encode_moco(self, res_dict, aug_imgs, num_crops):
        self._momentum_update_key_encoder()
        b = aug_imgs[0].size(0)
        res_dict['Z'], h, = [], []
        for i, aug_imgs_ in enumerate(aug_imgs):
            z_, h_ = self.encoder(aug_imgs_)[:2]
            res_dict['Z'].append(z_), h.append(F.normalize(h_, 2, 1))
            if num_crops[0] > i:
                with torch.no_grad():
                    self.mem_bank[i].append(F.normalize(self.encoder_k(aug_imgs_)[1], 2, 1))
                    if len(self.mem_bank[i]) > self.mem_bank_n:
                        self.mem_bank[i].pop(0)

        res_dict['loss'] = []
        for i, mem in enumerate(self.mem_bank):
            for j, hj in enumerate(h):
                if i != j:
                    mat = torch.matmul(hj, torch.cat(mem[::-1]).T) / self._t
                    res_dict['loss'].append(F.cross_entropy(mat, torch.arange(b).cuda()))

        res_dict['loss'] = torch.stack(res_dict['loss']).mean()
        res_dict['Z'] = torch.cat(res_dict['Z'])
        res_dict['class'] = self.evaluator(res_dict['Z'][:b * num_crops[0]])
        return res_dict

    def _encode_swav(self, res_dict, aug_imgs, num_crops):
        with torch.no_grad():
            w = self.prototypes.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.prototypes.weight.copy_(w)

        b = aug_imgs[0].size(0)
        res_dict['Z'], g = [], []

        for i, aug_imgs_ in enumerate(aug_imgs):
            z_, h_ = self.encoder(aug_imgs_)
            g.append(self.prototypes(h_))
            res_dict['Z'].append(z_)

        res_dict['loss'] = []
        for i, qi in enumerate(g[:num_crops[0]]):
            with torch.no_grad():
                qi = torch.exp(qi / self._e)
                qi = sinkhorn(qi.clone().T, 3)
            for j, qj in enumerate(g):
                if i != j:
                    qj = torch.log_softmax(qj / self._t, -1)
                    res_dict['loss'].append(-(qi * qj).sum(-1).mean())

        res_dict['loss'] = torch.stack(res_dict['loss']).mean()
        res_dict['Z'] = torch.cat(res_dict['Z'])
        res_dict['class'] = self.evaluator(res_dict['Z'][:b * num_crops[0]])
        return res_dict

    def _encode_byol(self, res_dict, aug_imgs, num_crops):
        self._momentum_update_key_encoder()
        res_dict['Z'], h, g, h_k = [], [], [], []
        b = aug_imgs[0].shape[0]
        for i, aug_imgs_ in enumerate(aug_imgs):
            z_, h_ = self.encoder(aug_imgs_)
            res_dict['Z'].append(z_), h.append(h_), g.append(self.predictor(h_))
            if num_crops[0] > i:
                with torch.no_grad():
                    h_k.append(self.encoder_k(aug_imgs_)[1])

        res_dict['loss'] = []
        for i, hi in enumerate(h_k):
            for j, hj in enumerate(g):
                if j != i:
                    res_dict['swav'].append(-F.cosine_similarity(hi.detach(), hj, dim=-1))

        res_dict['loss'] = torch.stack(res_dict['loss']).mean()
        res_dict['Z'] = torch.cat(res_dict['Z'])
        res_dict['class'] = self.evaluator(res_dict['Z'][:b * num_crops[0]])
        return res_dict

    def forward(self, x, class_only=False, nmb_crops=[2]):

        # dict for returning various values
        res_dict = {}
        if class_only:
            with torch.no_grad():
                z, h = self.encoder(x)
                res_dict['class'] = self.evaluator(z)
                res_dict['Z'] = torch.flatten(z, 1)
                res_dict['h'] = torch.flatten(h, 1)
                return res_dict

        if self.mem_bank is None:
            self.mem_bank = [[] for _ in range(nmb_crops[0])]

        if self.loss_type == 0:
            res_dict = self._encode_nce(res_dict, x, nmb_crops)
        elif self.loss_type == 1:
            res_dict = self._encode_moco(res_dict, x, nmb_crops)
        elif self.loss_type == 2:
            res_dict = self._encode_swav(res_dict, x, nmb_crops)
        elif self.loss_type == 3:
            res_dict = self._encode_byol(res_dict, x, nmb_crops)
        return res_dict
