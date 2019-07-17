import torch
import torch.nn as nn


def define_G(which_model_netG, norm ='batch', init_type = 'normal',gpu_ids = [], pretrain = True):
      
     netG       = None
     use_gpu    = len(gpu_ids) > 0
     norm_layer = get_norm_layer(norm_type = norm)

     if use_gpu:
        assert(torch.cuda.is_avaialable())
    
     netG       = ResnetX(id = 50, gpu_ids = gpu_ids , pretrain = pretrain)
  
     if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    
     if pretrain is True:
        print('Using pretrained weights')
   
     else
        print('Not using pretrained weights')
  
     init_weights(netG, init_type = init_type)

     return netG

def define_D(which_model_netD,norm = 'batch',use_sigmoid = False,init_type = 'normal', gpu_ids = []):
  
      netD       = None
      use_gpu    = len(gpu_ids) > 0
      norm_layer = get_norm_layer(norm_type = norm)
     
      if use_gpu:
         assert(torch.cuda.is_available())

      netD       = NLayerDiscriminator(4,64,n_layers = 3,norm_layer = norm_layer,use_sigmoid = use_sigmoid, gpu_ids = gpu_ids)

      if use_gpu:
         netD.cuda(gpu_ids[0])

      init_weights(netD,init_type = init_type)
      return netD


class GANLoss(nn.Module):

      def __init__(self,target_real_label = 1.0, target_fake_label = 0.0,tensor = torch.FloatTensor):
                
                  super(GANLoss,self).__init__()
                  self.real_label = target_real_label
                  self.fake_label = target_fake_label
                  self.fake_label_var = None
                  self.real_label_var = None

                  self.Tensor         = tensor

                  self.loss           = nn.BCELoss()

      def get_target_tensor(self,input,target_is_real):
                   
                   target_tensor   = None
                   if target_is_real:
                      create_label = ((self.real_label_var is None) or self.real_label_var.numel() !=input.numel())

                      if create_label :
                          real_tensor = self.Tensor(input.size()).fill(self.real_label)   
                          self.real_label_var = Variable(real_tensor,requires_grad = False)
                      target_tensor = self.real_label_var

                   else:
           
                      create_label = ((self.fake_label_var is None) or self.fake_label_var.numel() != input.numel())
                      
                      if create_label :
                           fake_tensor = self.Tensor(input.size()).fill_(self.real_label)
                           self.fake_label_var = Variable(fake_tensor,requires_grad = False)
                      target_tensor = self.fake_label_var
           
                    return target_tensor

class AlphaPredicitionLoss(nn.Module):
    
      def __init__(self):
            
                   super(AlphaPredictionLoss,self).__init__()

      def forward(self,input,target,trimap):
                 
                   #trimap region , 1 in unknown region, 0 in known regions
                   trimap_weights = torch.where(torch.eq(torch.ge(trimap,0.4),torch.le(trimap,0.6)),torch.ones_like(trimap),torch.ones_like(trimap))
                   unknown_region_size = trimap_weights.sum()
                   diff                = torch.sqrt(torch.add(torch.pow(input-target,2),1e-12))
                   return torch.mul(diff,trimap_weights).sum() / unknown_region_size


class CompLoss(nn.Module):
       
      def __init__(self):
             
                   super(CompLoss,self).__init__()

      def forward(self,input,target,trimap,fg,bg):
         
                   trimap_weights = torch.where(torch.eq(torch.ge(trimap,0.4),torch.le(trimap,0.6)),torch.ones_like(trimap),torch.ones_like(trimap))
                   unknown_region_size = trimap_weights.sum()

                   comp_target         = torch.mul(target,fg) + torch.mul((1.0 - target),bg)
                   comp_input          = torch.mul(input,fg) + torch.mul((1.0 - input),bg)

                   diff = torch.sqrt(torch.add(torch.pow(comp_input - comp_target, 2), 1e-12))
                   return torch.mul(diff, trimap_weights).sum() / unknown_region_size 


class ResNetX(nn.Module):
      
      def __init__(self,gpu_ids,pretrain):
     
                  super(ResNetX,self).__init__()
                  self.encoder  =   ResnetXEncoder(gpu_ids,pretrain)
                  self.decoder  =   ResnetXDecoder(gpu_ids)

      def forward(self,input):
 
                   #Encoder also gives us the saved pooling indices
                   x,ind   = self.encoder(input)
                   x       = self.decoder(x,ind)



class ResnetXEncoder(nn.Module):
      #Encoder has the same structure as that of ResNet50, but the last 2 layers are removed
      def __init__(self,pretrain):
         super(ResnetXEncoder,self).__init__()
         
         resnet   = models.resnet50(pretrained = pretrain)
         
         #Removing the last 2 Layers
         modules  = list(resnet.children())[:-2]

         #to save the pooling indices
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

        self.pool1  = nn.Sequential(*modules[: 4])
        self.resnet = nn.Sequential(*modules[4:])


        def forward(self,input):
        
            x,ind = self.pool1(input)
            x     = self.resnet(x)

            return x,ind



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

        init_weights(self.model1, 'xavier')
        init_weights(self.model2, 'xavier')

    def forward(self, input, ind):
        x = self.model1(input)
        x = self.unpool(x, ind)
        x = self.model2(x)

return x
            
                  

                      
      




