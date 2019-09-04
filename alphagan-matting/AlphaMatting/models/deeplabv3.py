import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True


__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, learn_bn=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        if not learn_bn:
            for i in self.bn1.parameters():
                i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        if not learn_bn:
            for i in self.bn2.parameters():
                i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        if not learn_bn:
            for i in self.bn3.parameters():
                i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ASPP_Module(nn.Module):
    def __init__(self, input_maps, dilation_series, padding_series, output_maps):
        super(ASPP_Module, self).__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(nn.Conv2d(input_maps, output_maps, kernel_size=1, stride=1, bias=False),
                                           nn.BatchNorm2d(output_maps, affine=affine_par)))

        for dilation, padding in zip(dilation_series, padding_series):
            self.branches.append(nn.Sequential(nn.Conv2d(input_maps, output_maps, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
                                               nn.BatchNorm2d(output_maps, affine=affine_par)))

        for m in self.branches:
            m[0].weight.data.normal_(0, 0.01)

        image_level_features = [nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(input_maps, output_maps, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(output_maps, affine=affine_par)]
        self.image_level_features = nn.Sequential(*image_level_features)
        self.conv1x1 = nn.Conv2d(output_maps*(len(dilation_series)+2), output_maps, kernel_size=1, stride=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(output_maps, affine=affine_par)

    def forward(self, x):
        out = self.branches[0](x)
        for i in range(len(self.branches)-1):
            out = torch.cat([out, self.branches[i+1](x)], 1)

        image_features = nn.functional.upsample(self.image_level_features(x), size=(out.shape[2],out.shape[3]), mode='bilinear')
        out = torch.cat([out, image_features], 1)
        out = self.conv1x1(out)
        out = self.bn1x1(out)

        return out


class Encoder(nn.Module):
    def __init__(self, block, layers, output_maps=1024, learn_bn=True):
        self.inplanes = 64
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        if not learn_bn:
            for i in self.bn1.parameters():
                i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, learn_bn=learn_bn)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, learn_bn=learn_bn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, learn_bn=learn_bn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, learn_bn=learn_bn)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=1, learn_bn=learn_bn)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=1, learn_bn=learn_bn)
        self.layer5 = ASPP_Module(2048, [6,12,18],[6,12,18], output_maps)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        
        # Change input shape of the first convolutional layer
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        weights = torch.zeros(64, 4, 7, 7)
        weights[:,:3,:,:] = self.conv1.weight.data.view(64, 3, 7, 7)
        conv1.weight.data.copy_(weights)
        self.conv1 = conv1

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, learn_bn=True, multi_grid=(1,1,1)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        if not learn_bn:
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=int(dilation*multi_grid[0]), downsample=downsample, learn_bn=learn_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=int(dilation*multi_grid[i%3]), learn_bn=learn_bn))

        return nn.Sequential(*layers)


    def forward(self, x):
        skip1 = x[:,:3,:,:]
        x = self.conv1(x)
        x = self.bn1(x)
        skip2 = self.relu(x)
        x, ind = self.maxpool(skip2)

        skip3 = self.layer1(x)
        x = self.layer2(skip3)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x, ind, skip1, skip2, skip3


class Decoder(nn.Module):
    def __init__(self, feature_size=40):
        super(Decoder, self).__init__()
        model = [nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                 nn.BatchNorm2d(1024),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(1024, 1024, kernel_size=1, stride=2, output_padding=1, bias=False),
                 nn.BatchNorm2d(1024),
                 nn.ReLU(True),
                 nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                 nn.BatchNorm2d(1024),
                 nn.ReLU(True)]
        self.upsample1 = nn.Sequential(*model)

        model = [nn.Conv2d(1280, 1024, kernel_size=3, padding=1),
                 nn.BatchNorm2d(1024),
                 nn.ReLU(True),
                 nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                 nn.BatchNorm2d(512),
                 nn.ReLU(True),
                 nn.Conv2d(512, 256, kernel_size=3, padding=1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(True),
                 nn.Conv2d(256, 64, kernel_size=3, padding=1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True)]
        self.upsample2 = nn.Sequential(*model)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv1x1_2 = nn.Conv2d(64, 16, kernel_size=1)

        model = [nn.Conv2d(80, 64, kernel_size=3, padding=1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True)]
        self.upsample3 = nn.Sequential(*model)

        model = [nn.Conv2d(67, 64, kernel_size=3, padding=1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 nn.Conv2d(64, 1, kernel_size=3, padding=1),
                 nn.Sigmoid()]
        self.upsample4 = nn.Sequential(*model)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input, ind, skip1, skip2, skip3):
        x = self.upsample1(input)
        skip = skip3
        x = torch.cat([x, skip], 1)
        x = self.upsample2(x)
        x = self.unpool(x, ind)
        skip = self.conv1x1_2(skip2)
        x = torch.cat([x, skip], 1)
        x = self.upsample3(x)
        x = torch.cat([x, skip1], 1)
        x = self.upsample4(x)

        return x


class ResNet(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(ResNet, self).__init__()
        self.gpu_ids = gpu_ids

        self.encoder = Encoder(Bottleneck, [3, 4, 6, 3])
        self.decoder = Decoder()

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            x, ind, skip1, skip2, skip3 = nn.parallel.data_parallel(self.encoder, input, self.gpu_ids)
            out = nn.parallel.data_parallel(self.decoder, (x, ind, skip1, skip2, skip3), self.gpu_ids)

            return out
        else:
            x, ind, skip1, skip2, skip3 = self.encoder(input)
            out = self.decoder(x, ind, skip1, skip2, skip3)

            return out
