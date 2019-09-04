from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch
from PIL import Image


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)
        self.netG = self.netG.eval()
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        #we need to use single_dataset mode
        A_fg     = input['A_fg']
        A_trimap = input['A_trimap']
        
        if len(self.gpu_ids) > 0:
             A_fg     = A_fg.cuda(self.gpu_ids[0],async=True)
             A_trimap = A_trimap.cuda(self.gpu_ids[0],async=True)
  
        self.trimap_A    = A_trimap
        self.fg_A        = A_fg         
        self.input_A     = torch.cat((self.fg_A,self.trimap_A),1)
        self.image_paths = input['A_paths']

    def forward(self,input):
        A_fg     = input['A_fg']
        A_trimap = input['A_trimap']
        
        self.w        = input['w']
        self.h        = input['h']
        
        if len(self.gpu_ids) > 0:
             A_fg     = A_fg.cuda(self.gpu_ids[0],async=True)
             A_trimap = A_trimap.cuda(self.gpu_ids[0],async=True)
  
        self.trimap_A    = A_trimap
        self.fg_A        = A_fg         
        self.input_A     = torch.cat((self.fg_A,self.trimap_A),1)
        self.image_paths = input['A_paths']

        #self.real_A  = Variable(self.fg_A)
        #self.real_A      = self.real_A_320.resize((w,h))
        self.fake_B  = self.trimap_merge(self.netG(self.input_A),self.trimap_A)
        #self.fake_B      = self.fake_B_320.resize((w,h))

        


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        #real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('fake_B', fake_B)])

    def trimap_merge(self, alpha, trimap):
        
        # Using the already known regions from trimap
        final_alpha = torch.where(torch.eq(torch.ge(trimap, 0.4), torch.le(trimap, 0.6)), alpha, trimap)
        return final_alpha

