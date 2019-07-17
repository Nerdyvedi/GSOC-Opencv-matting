import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from image_pool import ImagePool



class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class Model(BaseModel):
     def name(self):
         return 'Model'
   
     def initialize(self,opt):
         BaseModel.initialize(self,opt)
         nb    = opt.batchSize
         size  = opt.fineSize
         
         self.netG = define_G(4,1,opt.ngf,opt.init_type,self.gpu_ids)
         self.netD = define_D(4,opt.ndf,opt.init_type,self.gpu_ids)

         if opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG,'G',opt.which_epoch)
            self.load_network(self.netD,'D',opt.which_epoch)

         self.old_lr          = opt.lr
         self.criterion_alpha = losses.AlphaPredictionLoss()
         self.criterionComp   = networks.compLoss
         self.criterionGAN    = networks.GANLoss(tensor = self.Tensor)
         self.fake_A_pool     = ImagePool(opt.pool_size)

         #Initializing the optimzers
         self.optimizer_G     = torch.adam.optimizer(self.netG.parameters(),lr = opt.lr, betas = (0.9,0.999))
         self.optimizer_D     = torch.adam.optimizer(self.netD.parameters(),lr = opt.lr, betas = (0.9,0.999))
        
         self.optimizers      = []
         self.optimizers.append(optimizer_G)
         self.optimizers.append(optimizer_D)
     
         self.schedulers      = []
         # Get learning rate schedulers for both optimizers

         for optimizer in self.optimizers:
              self.schedulers.append(networks.get_scheduler(optimizer,opt))
         
         print('--------------------Network Initialized-------------------')
         

     def set_input(self,opt):
         A_bg     = input['A_bg']
         A_fg     = input['A_fg']
         A_alpha  = input['A_alpha']
         A_trimap = input['A_trimap']

         if(len(self.gpu_ids > 0)):
            A_bg       =  A_bg.cuda(self.gpu_ids[0], async = True)
            A_fg       =  A_fg.cuda(self.gpu_ids[0], async = True)
            A_alpha    =  A_alpha.cuda(self.gpu_ids[0], async = True)
            A_trimap   =  A_trimap.cuda(self.gput_ids[0], async = True)

         
         self.bg     = A_bg
         self.fg     = A_fg
         self.alpha  = A_alpha
         self.trimap = A_trimap
         self.img    = self.composite(self.alpha,self.fg,self.bg)
       
     def set_input_predict(self,opt):
          A_img      = input['A_img']
          A_trimap   = input['A_trimap']
      
          if(len(self.gpu_ids > 0)):
            A_img    = A_img.cuda(self.gpu_ids[0], async = True)
            A_trimap = A_trimap.cuda(self.gpu_ids[0], async = True)
 
          self.A_trimap = A_trimap
          self.A_img      = A_img

          self.input_A  = torch.cat((self.A_img,self.trimap),1)  

     def trimap_merge(self,alpha,trimap):
          
         #Those regions of which we are sure of are added to alpha
         final_alpha = torch.where(torch.eq(torch.ge(trimap,0.4),torch.le(trimap,0.6)),alpha,trimap)

     def composite(self,alpha,fg,bg):
         
          img = torch.mul(alpha,fg) + torch.mul(1-alpha,bg)
          return img

     def forward(self):
         
          self.A_input  = Variable(torch.cat((self.img,self.trimap),1))
          self.A_fg     = Variable(self.fg)
          self.A_trimap = Variable(self.trimap)
          self.A_bg     = Variable(self.bg)
          self.A_img    = Variable(self.img)
          self.A_alpha  = Variable(self.alpha)

     def predict(self):
        
          self.netG.eval()
          with torch.no_grad():
               self.real         = Variable(self.A_img)
               self.fake_alpha   = self.netG(Variable(self.A_input))
               self.trimap_A     = Variable(self.A_trimap)
               
     def backward_D_basic(self,netD,real,fake):
          
          #Discriminator should identify real as real
          pred_real    = netD(real)
          loss_D_real  = self.criterionGAN(pred_real,True)

          #Discriminator should identify fake as fake
          pred_fake    = netD(fake)
          loss_D_fake  = self.criterionGAN(pred_fake,False)

          #Total loss is the sum
          loss_D       = loss_D_real + loss_D_fake
          
          loss_D.backward()
          
          return loss_D

     def backward_D(self):
         
          #This is used to ensure discriminator also takes previously generated images into account as well
          fake_comp = self.fake_A_pool.query(self.comp_disc)
          loss_D    = self.backward_D_basic(self.netD, self.A_input, fake_comp)
          
          self.loss_D = loss_D.data[0]
          
   
     def backward_G(self):
          
          pred = self.netG(self.A_input)
          pred = self.trimap_merge(pred,self.A_trimap)
          #Image formed by predicition
          comp = self.composite(pred,self.A_fg,self.A_bg)
          
          #Input to the discriminator is the image and the trimap
          comp_disc = torch.cat((comp,self.A_trimap),1)
          
          pred_fake = self.netD(comp_disc)
 
          #Generator wants fake data to be percieved as true by the discriminator 
          loss_g    = self.criterionGAN(pred_fake,True)

          #Alpha prediction loss
          loss_a    = self.criterionAlpha(pred,self.A_alpha,self.A_trimap)

          loss_c    = 0
         
          loss      = loss_a + loss_g + loss_c
    
          loss.backward()

          self.pred      = pred.data
          self.comp_disc = comp_disc.data
  
          self.loss_a    = loss_a.data[0]
          self.loss_c    = 0
          self.loss_g	 = loss_g.data[0]                  
        
          
     
     def optimize_parameters(self):

          self.forward()

          self.optimizer_G.zero_grad()
          self.backward_G()
          self.optimizer_G.step()
         
          self.optimizer_F.zero_grad()
          self.backward_D()
          self.optimizer_D.step()
          
 

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model       = Model()
visualizer  = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)

        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
model.update_learning_rate()
