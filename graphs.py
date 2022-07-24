import numpy as np
from torch import nn
import torch
from torch.autograd import Variable

'''Modify version of Pytorch's ResNet'''


class ResNet(nn.Module):
    def __init__(self, block, layers, conv, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, bn_mom=0.1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.block = block
        self._norm_layer = norm_layer
        self.bn_mom = bn_mom
        self.covn1 = conv
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])  # 16x16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        self.layer_list = nn.ModuleList([self.covn1, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool])
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, momentum=self.bn_mom),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        layer_acts = [x]
        for i, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)
        return layer_acts  # [-1]

    def forward(self, x):
        return self._forward_impl(x)


class Projection(nn.Module):
    def __init__(self, n_input, n_out=128, n_hidden=512, batch=True, h=1):
        super(Projection, self).__init__()
        # '''

        layers = []
        for i in range(h):
            layers.append(nn.Linear(n_input, n_hidden, bias=False))
            if batch:
                layers.append(nn.BatchNorm1d(n_hidden, affine=True))
            layers.append(nn.ReLU(inplace=True))
            n_input = n_hidden
        layers.append(nn.Linear(n_input, n_out, bias=False))
        self.project = nn.Sequential(*layers)

        '''
        self.project = nn.Sequential(SpectralNorm(nn.Linear(n_input, n_hidden)),
                                     #nn.BatchNorm1d(n_hidden, affine=True),
                                     nn.ReLU(inplace=True),
                                     # SpectralNorm(nn.Linear(n_hidden, n_hidden)),
                                     # nn.BatchNorm1d(n_hidden, affine=True),
                                     # nn.ReLU(inplace=True),
                                     SpectralNorm(nn.Linear(n_hidden, n_out)))
        # '''

        '''
        if batch is False:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                    # nn.init.uniform_(m.weight, 0, 1 / np.sqrt(m.weight.shape[1]))
                    nn.init.normal_(m.weight, 0, 1. / np.sqrt(m.weight.shape[1]))
                    # nn.init.normal_(m.weight, 0, 0.1)
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
                    pass
                # elif isinstance(m, nn.BatchNorm1d):
                #    nn.init.constant_(m.weight, 1)
                #    nn.init.constant_(m.bias, 0)
        # '''
        '''
        self.u = []
        for layer in self.project.modules():
            if layer._get_name() == 'Linear':
                self.u.append(torch.randn((1, layer.weight.shape[-1]), device='cuda', requires_grad=False))
                pass
        #'''
        return

    @staticmethod
    def spectral_normed_weight(w, u, ip=1):
        u_ = u
        for _ in range(ip):
            v_ = torch.nn.functional.normalize(torch.matmul(u_, w.T), 2, 1)
            u_ = torch.nn.functional.normalize(torch.matmul(v_, w), 2, 1)
        sigma = torch.matmul(torch.matmul(v_, w), u_.T).flatten(0)
        w /= sigma
        return w, sigma, u_

    @torch.no_grad()
    def sn_layer(self, ip=1):
        i = 0
        for layer in self.project.modules():
            if type(layer) == nn.Linear:
                w = layer.weight.data
                w, sigma, self.u[i] = self.spectral_normed_weight(w, self.u[i], ip)
                layer.weight.copy_(w)
                i += 1

    def get_weight(self):
        if isinstance(self.project[0], nn.Linear):
            return self.project[0].weight
        else:
            return self.project[0].module.weight_bar

    def init_param(self):
        for m in self.project.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                m.bias.data.fill_(0)
                m.weight.data.copy_(torch.randn_like(m.weight.data) * 0.02)

    def forward(self, r1_x):
        return self.project(r1_x)


class ProjectionCNN(nn.Module):
    def __init__(self, n_input, n_out=128, n_hidden=512, batch=True, h=1, k=1):
        super(ProjectionCNN, self).__init__()
        # '''

        layers = []
        for i in range(h):
            layers.append(nn.Conv2d(n_input, n_hidden, k, 1, bias=False))
            if batch:
                layers.append(nn.BatchNorm2d(n_hidden, affine=True))
            layers.append(nn.ReLU(inplace=True))
            n_input = n_hidden
            k = 1

        layers.append(nn.Conv2d(n_hidden, n_out, 1, 1, bias=False))
        self.project = nn.Sequential(*layers)

        return

    def get_weight(self):
        if isinstance(self.project[0], nn.Linear):
            return self.project[0].weight
        else:
            return self.project[0].module.weight_bar

    def forward(self, r1_x):
        return self.project(r1_x)


class Predictor(nn.Module):
    def __init__(self, n_input=128, n_out=128, n_hidden=512, batch=True, h=0):
        super(Predictor, self).__init__()

        layers = []
        for i in range(h):
            layers.append(nn.Linear(n_input, n_hidden, bias=False))
            if batch:
                layers.append(nn.BatchNorm1d(n_hidden, affine=True))
            layers.append(nn.ReLU(inplace=True))
            n_input = n_hidden

        layers.append(nn.Linear(n_hidden, n_out, bias=False))
        self.project = nn.Sequential(*layers)

    def forward(self, x):
        return self.project(x)


class MLPClassifier(nn.Module):
    def __init__(self, n_classes, n_input, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes

        self.block_forward = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(self.n_input, n_classes, bias=True)
        )

    def forward(self, x):
        with torch.no_grad():
            x_ = x.detach()
            x_ = torch.flatten(x_, 1)
        # x_ = torch.flatten(x, 1)
        logits = self.block_forward(x_)
        return logits


class Prototypes(nn.Module):
    def __init__(self, n_input, n_out=1000):
        super(Prototypes, self).__init__()
        self.prototypes = nn.Linear(n_input, n_out, bias=False)
        return

    def forward(self, r1_x):
        r1_x = nn.functional.normalize(r1_x, dim=1, p=2)
        p = self.prototypes(r1_x)
        return p


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        w = getattr(self.module, self.name + "_bar")
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        # print(u.dtype, w.dtype)
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data)).type_as(w)
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data)).type_as(w)

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v)).type_as(w)
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


