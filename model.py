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
        rkhs_4 = self.model(dummy_batch)[-3]
        self.emb_dim = rkhs_1.size(1)
        self.emb_dim_4 = rkhs_4.size(1)
        self.project = Projection(self.emb_dim, project_dim, self.emb_dim, batch=batch, h=h)
        # self.project = Projection(self.emb_dim, project_dim, 1024, batch=batch, h=h)
        # self.project = ProjectionCNN(self.emb_dim, project_dim, 1024, batch=batch, h=h, k=1)
        # self.project_4 = ProjectionCNN(self.emb_dim_4, project_dim, 1024, batch=batch, h=h, k=3)

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
    def __init__(self, n_classes, encoder_size=32, prototypes=1000, project_dim=128,
                 temp=0.1, eps=0.05, epoch=500, mom=0.99, loss_type=0,
                 model_type='resnet18', batch_mlp=True, hidden_n=1, bank_n=10, logger=None):
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
            'bank_n': bank_n,
            'logger': logger
        }
        self.logger = logger

        self.encoder = Encoder(encoder_size=encoder_size, project_dim=project_dim, model_type=model_type,
                               batch=batch_mlp, h=hidden_n)
        self.encoder_k = Encoder(encoder_size=encoder_size, project_dim=project_dim, model_type=model_type,
                                 batch=batch_mlp, h=hidden_n)
        self.prototypes = Prototypes(project_dim, prototypes)
        self.prototypes_k = Prototypes(project_dim, prototypes)
        self.predictor = Projection(project_dim, project_dim, h=hidden_n, batch=batch_mlp)
        self.predictor2 = Predictor(project_dim * 2, project_dim, h=3)
        # self.predictor3 = Projection(project_dim, project_dim)
        # self.predictor3_k = Projection(project_dim, project_dim)
        self.C = 512
        self.evaluator = MLPClassifier(n_classes, self.encoder.emb_dim)
        self.rbf = rbf
        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False
        for param, param_k in zip(self.prototypes.parameters(), self.prototypes_k.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False

        self.info_modules = [self.encoder.project, self.predictor, self.predictor2]
        # self.predictor2, self.predictor3],
        # self.info_modules = [self.encoder, self.prototypes, self.predictor,
        #                     self.predictor2, self.predictor3]
        self.class_modules = [self.evaluator]
        self._t, self._e = temp, eps
        self.m, self.loss_type = mom, loss_type
        self.step = 0
        self.epoch = epoch
        self.i = 0
        self.k_scale = k_scale
        self.mem_bank = None
        self.mem_bank_m = None
        self.mem = bank_n
        self.mu = torch.normal(0, 1, (n_classes, project_dim)).cuda()

    def get_details(self, lam):
        b = 'b' if self.hyperparams['batch_mlp'] else 'nb'
        return str(self.hyperparams['model_type']) + '_' + str(int(self.hyperparams['loss_type'])) + '_' \
               + str(self.hyperparams['hidden_n']) + '_' + b + '_' + str(self.hyperparams['project_dim']) + '_' \
               + str(self.hyperparams['mom']) + '_' + str(np.around(lam, 2))

    def encode(self, x, use_eval=False):
        '''
        Encode the images in x, with or without grads detached.
        '''
        if use_eval:
            self.eval()
        rkhs_1, rkhs_1_ = self.encoder(x)[:2]
        if use_eval:
            self.train()
        return rkhs_1, rkhs_1_

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        m = self.m  # 1 - (1 - self.m) * (np.cos(self.step * np.pi / self.epoch) + 1) / 2
        # m = 1 - (1 - self.m) * (np.cos(self.step * np.pi / self.epoch) + 1) / 2
        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param.data * (1 - m) + param_k.data * m
        for param, param_k in zip(self.prototypes.parameters(), self.prototypes_k.parameters()):
            param_k.data = param.data * (1 - m) + param_k.data * m

    @torch.no_grad()
    def _print_stat(self, h, h2):
        self.logger.info((np.around(h.mean().item(), 2), np.around(h.std(1).mean().item(), 2),
                          np.around(h.std(0).mean().item(), 2), np.around(h.std(1).std().item(), 2),
                          np.around(h.std(0).std().item(), 2)))
        with torch.no_grad():
            aa = torch.unsqueeze((h * h).sum(-1), 1)
            bb = torch.unsqueeze((h2 * h2).sum(-1), 0)
            mm = aa + bb - 2 * torch.mm(h, h2.T)
            score = mm.argmin(-1) == torch.arange(mm.shape[0], device='cuda')
            self.logger.info(('mm', np.around(score.sum().item() / float(mm.shape[0]), 2)))

    def _encode_nce(self, res_dict, aug_imgs, num_crops):

        b = aug_imgs[0].size(0)
        res_dict['Z'], h = [], []
        tmp = torch.tensor([0]).cuda()

        for i, aug_imgs_ in enumerate(aug_imgs):
            z_, h_ = self.encoder(aug_imgs_)[:2]
            res_dict['Z'].append(z_), h.append(h_)

        res_dict['loss'], res_dict['swav'] = [], []
        h_normed = [F.normalize(hi, 2, 1) for hi in h]

        mat = torch.matmul(h_normed[0], torch.cat(h_normed).T) / self._t
        mask_pos = torch.eye(b, dtype=torch.bool, device='cuda').repeat(1, np.sum(num_crops))
        mask_neg = torch.logical_xor(torch.tensor([True]).cuda(), mask_pos)
        pos = mat.masked_select(mask_pos).view(b, -1)[:, 1:]
        neg = mat.masked_select(mask_neg).view(b, -1).exp().sum(-1, keepdims=True)
        res_dict['swav'] = -(pos - torch.log(neg + pos.exp())).mean()

        self.i += 1
        if self.i % 200 == 0:
            self._print_stat(res_dict['Z'][0], res_dict['Z'][1])
            self._print_stat(h[0], h[1])

        res_dict['norm'] = tmp
        res_dict['loss'] = torch.stack(res_dict['loss']).mean() if len(res_dict['loss']) > 0 else tmp
        res_dict['Z'] = torch.cat(res_dict['Z'])
        res_dict['class'] = self.evaluator(res_dict['Z'][:b * num_crops[0]])

        return res_dict

    def _encode_moco(self, res_dict, aug_imgs, num_crops):
        self._momentum_update_key_encoder()
        b = aug_imgs[0].size(0)
        res_dict['Z'], h, h_k = [], [], []
        tmp = torch.tensor([0]).cuda()

        for i, aug_imgs_ in enumerate(aug_imgs):
            z_, h_ = self.encoder(aug_imgs_)[:2]
            res_dict['Z'].append(z_), h.append(h_)
            if num_crops[0] > i:
                with torch.no_grad():
                    self.mem_bank[i].append(F.normalize(self.encoder_k(aug_imgs_)[1], 2, 1))
                    if len(self.mem_bank[i]) > self.mem:
                        self.mem_bank[i].pop(0)

        res_dict['loss'], res_dict['swav'] = [], []
        h_normed = [F.normalize(hi, 2, 1) for hi in h]

        for i, mem in enumerate(self.mem_bank):
            for j, hj in enumerate(h_normed):
                if i != j:
                    mat = torch.matmul(hj, torch.cat(mem[::-1]).T) / self._t
                    res_dict['swav'].append(F.cross_entropy(mat, torch.arange(b).cuda()))

        self.i += 1
        if self.i % 200 == 0:
            self._print_stat(res_dict['Z'][0], res_dict['Z'][1])
            self._print_stat(h[0], h[1])

        res_dict['norm'] = tmp
        res_dict['loss'] = torch.stack(res_dict['loss']).mean() if len(res_dict['loss']) > 0 else tmp
        res_dict['swav'] = torch.stack(res_dict['swav']).mean() if len(res_dict['swav']) > 0 else tmp
        res_dict['Z'] = torch.cat(res_dict['Z'])
        res_dict['class'] = self.evaluator(res_dict['Z'][:b * num_crops[0]])

        return res_dict

    def _encode_swav(self, res_dict, aug_imgs, num_crops):

        with torch.no_grad():
            w = self.prototypes.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.prototypes.weight.copy_(w)

        b = aug_imgs[0].size(0)
        res_dict['Z'], h, g = [], [], []
        tmp = torch.tensor([0]).cuda()
        for i, aug_imgs_ in enumerate(aug_imgs):
            z_, h_ = self.encoder(aug_imgs_)[:2]
            g.append(self.prototypes(h_))
            res_dict['Z'].append(z_), h.append(h_)

        res_dict['loss'], res_dict['swav'] = [], []

        for i, qi in enumerate(g[:num_crops[0]]):
            with torch.no_grad():
                qi = torch.exp(qi / self._e)
                qi = sinkhorn(qi.clone().T, 3)
            for j, qj in enumerate(g):
                if i != j:
                    qj = torch.log_softmax(qj / self._t, -1)
                    res_dict['swav'].append(-(qi * qj).sum(-1).mean())

        self.i += 1
        if self.i % 200 == 0:
            self._print_stat(res_dict['Z'][0], res_dict['Z'][1])
            self._print_stat(h[0], h[1])

        res_dict['norm'] = tmp
        res_dict['swav'] = torch.stack(res_dict['swav']).mean() if len(res_dict['swav']) > 0 else tmp
        res_dict['loss'] = torch.stack(res_dict['loss']).mean() if len(res_dict['loss']) > 0 else tmp
        res_dict['Z'] = torch.cat(res_dict['Z'])
        res_dict['class'] = self.evaluator(res_dict['Z'][:b * num_crops[0]])

        return res_dict

    def _encode_byol(self, res_dict, aug_imgs, num_crops):
        res_dict['Z'], h, g, h_k = [], [], [], []
        tmp = torch.tensor([0]).cuda()
        res_dict['loss'], res_dict['swav'], res_dict['norm'] = [], [], []
        b = aug_imgs[0].shape[0]
        for i, aug_imgs_ in enumerate(aug_imgs):
            z_, h_ = self.encoder(aug_imgs_)[:2]
            res_dict['Z'].append(z_), h.append(h_), g.append(self.predictor(h_))

        for i, hi in enumerate(h_k):
            for j, hj in enumerate(g):
                if j != i:
                    res_dict['swav'].append(-F.cosine_similarity(hi.detach(), hj, dim=-1))

        res_dict['norm'] = torch.stack(res_dict['norm']).mean() if len(res_dict['norm']) > 0 else tmp
        res_dict['swav'] = torch.stack(res_dict['swav']).mean() if len(res_dict['swav']) > 0 else tmp
        res_dict['loss'] = torch.stack(res_dict['loss']).mean() if len(res_dict['loss']) > 0 else tmp
        res_dict['Z'] = torch.cat(res_dict['Z'])
        res_dict['class'] = self.evaluator(res_dict['Z'][:b * num_crops[0]])

        return res_dict

    def forward(self, x, class_only=False, nmb_crops=[2], idxs=None):

        # dict for returning various values
        res_dict = {}
        if class_only:
            with torch.no_grad():
                rkhs_1, rkhs_1_ = self.encode(x)
                res_dict['class'] = self.evaluator(rkhs_1)
                res_dict['rkhs_glb'] = torch.flatten(rkhs_1_, 1)
                res_dict['emb'] = torch.flatten(rkhs_1, 1)
                res_dict['emb_'] = torch.flatten(rkhs_1_, 1)
                return res_dict

        self._momentum_update_key_encoder()

        if self.mem_bank is None:
            self.mem_bank = [[] for _ in range(nmb_crops[0])]

        if self.loss_type == 0:
            res_dict = self._encode_nce(res_dict, x, nmb_crops)
        elif self.loss_type == 1:
            res_dict = self._encode_moco(res_dict, x, nmb_crops)
        elif self.loss_type == 2:
            res_dict = self._encode_swav(res_dict, x, nmb_crops)
        elif self.loss_type == 3:
            res_dict = self._encode_pred_mom_sg(res_dict, x, nmb_crops)
        return res_dict
