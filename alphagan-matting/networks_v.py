import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
from . import deeplabv3
###############################################################################


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

    

def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], pretrain=True):
    netG = None
    use_gpu    = len(gpu_ids) > 0
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)


    if use_gpu:
        assert(torch.cuda.is_available())

    netG = ResnetX(id=50, gpu_ids=gpu_ids, pretrain=pretrain)
    
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    if (which_model_netG == 'resnet50' or which_model_netG == 'deeplabv3') and pretrain:
        print('Using pretrained weights')
    else:
        print('Not using pretrained weights')
        netG.apply(weights_init_xavier)
    
    return netG


def define_D(input_nc,ndf, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu    = len(gpu_ids) > 0
    norm_layer = functools.partial(nn.BatchNorm2d, affine = True)

    if use_gpu :
       assert(torch.cuda.is_available())

    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    
    netD.apply(weights_init_xavier)
    return netD


#There are 3 types of losses, adversial loss, compositional loss and alpha prediction loss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input)


class AlphaPredictionLoss(nn.Module):
    def __init__(self):
        super(AlphaPredictionLoss, self).__init__()

    def forward(self, input, target, trimap):
        trimap_weights = torch.where(torch.eq(torch.ge(trimap, 0.4), torch.le(trimap, 0.6)), torch.ones_like(trimap), torch.zeros_like(trimap))
        unknown_region_size = trimap_weights.sum()
        diff = torch.sqrt(torch.add(torch.pow(input - target, 2), 1e-12))
        return torch.mul(diff, trimap_weights).sum() / unknown_region_size

class CompLoss(nn.Module):
    def __init__(self):
        super(CompLoss, self).__init__()

    def forward(self, input, target, trimap, fg, bg):
        trimap_weights = torch.where(torch.eq(torch.ge(trimap, 0.4), torch.le(trimap, 0.6)), torch.ones_like(trimap), torch.zeros_like(trimap))
        unknown_region_size = trimap_weights.sum()

        comp_target = torch.mul(target, fg) + torch.mul((1.0 - target), bg)       
        comp_input = torch.mul(input, fg) + torch.mul((1.0 - input), bg)

        diff = torch.sqrt(torch.add(torch.pow(comp_input - comp_target, 2), 1e-12))
        return torch.mul(diff, trimap_weights).sum() / unknown_region_size




class ResnetX(nn.Module):
    def __init__(self, id=50, gpu_ids=[], pretrain=True):
        super(ResnetX, self).__init__()
        self.encoder = ResnetXEncoder(id, gpu_ids, pretrain)
        self.decoder = ResnetXDecoder(gpu_ids)      

    def forward(self, input):
        x, ind = self.encoder(input)
        x = self.decoder(x, ind)

        return x


class ResnetXEncoder(nn.Module):
    #Encoder has the same structure as that of Resnet50,but the last 2 layers have been removed
    def __init__(self, id=50, pretrain=True, gpu_ids=[]):
        super(ResnetXEncoder, self).__init__()
        print('Pretrain: {}'.format(pretrain))
        if id==50:
            resnet = models.resnet50(pretrained=pretrain)
        elif id==101:
            resnet = models.resnet101(pretrained=pretrain)
        elif id==34:
            resnet = models.resnet34(pretrained=pretrain)
        modules = list(resnet.children())[:-2] # delete the last 2 layers.
        for m in modules:
            if 'MaxPool' in m.__class__.__name__:
                m.return_indices = True

        # Change input shape of the first convolutional layer
        # Resnet had 3 channels, but for this task we need 4 channels, as we are also adding the trimap
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        weights = torch.zeros(64, 4, 7, 7)
        weights[:,:3,:,:] = modules[0].weight.data.view(64, 3, 7, 7)
        conv1.weight.data.copy_(weights)
        modules[0] = conv1

        self.pool1 = nn.Sequential(*modules[: 4])
        self.resnet = nn.Sequential(*modules[4:])

    def forward(self, input):
        x, ind = self.pool1(input)
        x = self.resnet(x)

        return x, ind

class ResnetXDecoder(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(ResnetXDecoder, self).__init__()
        model = [nn.Conv2d(2048, 2048, kernel_size=1, padding=0),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(2048, 1024, kernel_size=1, stride=2, output_padding=1, bias=False),
                 # nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                 nn.BatchNorm2d(1024),
                 nn.ReLU(True)]
        model += [nn.Conv2d(1024, 1024, kernel_size=5, padding=2),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2, output_padding=1, bias=False),
                  # nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                  nn.BatchNorm2d(512),
                  nn.ReLU(True)]
        model += [nn.Conv2d(512, 512, kernel_size=5, padding=2),
                  nn.BatchNorm2d(512),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2, output_padding=1, bias=False),
                  # nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=5, padding=2),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True),
                  nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)]  
        model += [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)]
        self.model1 = nn.Sequential(*model)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        model = [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                  nn.Conv2d(64, 1, kernel_size=5, padding=2),
                  nn.Sigmoid()]
        self.model2 = nn.Sequential(*model)

        model1.apply(weights_init_xavier)
        model2.apply(weights_init_xavier)s

    def forward(self, input, ind):
        x = self.model1(input)
        x = self.unpool(x, ind)
        x = self.model2(x)

        return x



class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
           return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
           return self.model(input)









