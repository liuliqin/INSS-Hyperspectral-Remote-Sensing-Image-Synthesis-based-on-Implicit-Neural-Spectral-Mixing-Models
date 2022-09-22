import torch,os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .generation_loss import GenerationLoss
import sys
sys.path.append('..')
from utils.util import batch_PSNR,Logger,write_img
from metric.calcul_metric import Cal
from networks.Abunreger import Abunreger
from networks.Reslearner_nonlinear import Reslearner
from networks.Discriminators import spe_discriminator,spa_discriminator
import torch.optim as optim
import scipy.io as scio

class AvirisGeneration(nn.Module):
    def __init__(self,spe_lib,opt,ini_saa,mode ='Train',weight_band = torch.ones([1,224])):
        super(AvirisGeneration,self).__init__()
        self.device  = opt.device

        self.num_class = spe_lib.size()[1]
        self.hyper_channel = spe_lib.size()[0]
        self.multi_channel = opt.input_channel

        self.ini_saa = ini_saa
        self.spe_lib = spe_lib

        self.root = os.path.join(opt.save_root,opt.dataset,opt.identy_root)
        self.model_root = os.path.join(self.root,'model')
        if not os.path.isdir(self.root):
            os.mkdir(self.root)
        if not os.path.isdir(self.model_root):
            os.mkdir(self.model_root)

        self.opt = opt
        self.mode = mode

        # removal abnormal bands when calculate loss
        close_list = list(range(103)) + list(range(114, 151)) + list(range(168, 224))
        self.close_list = close_list
        self.cal_channel = len(self.close_list)
        if mode is 'Train': # training loss functions
            self.loss_functions = GenerationLoss(opt,self.hyper_channel,weight_band,close_list)
        else:  #calculate metrics
            self.save_root = os.path.join(self.root,'test_results_'+opt.model_epoch)
            if not os.path.isdir(self.save_root):
                os.mkdir(self.save_root)
            self.cal_performance = opt.cal_performance
            if self.cal_performance:
                self.cal_performance = Cal(65535,close_list)
            self.abun_flag = opt.abun_flag# whether save abundance

    def build_networks(self):
        self.Abunreger= Abunreger(self.ini_saa, self.opt.ngf, self.multi_channel, self.num_class,self.opt.downscale_factor + 1)  # abundance prediction network F
        self.Reslearner= Reslearner(self.opt, self.hyper_channel)  # feature transform network T and nerual mixing network N
        # weight initial
        self.Abunreger.weight_init(mean=0.0, std=0.02)
        self.Reslearner.weight_init()
        ## to device
        self.Abunreger.to(self.device)
        self.Reslearner.to(self.device)
        # discriminators
        if self.mode is 'Train':
            self.D_spa = spa_discriminator(self.opt.ndf, self.cal_channel+self.multi_channel)
            self.D_spe = spe_discriminator(self.cal_channel, self.opt.spe_inter_size, 2 * self.opt.ndf)
            # weight initial
            self.D_spa.weight_init(mean=0.0, std=0.02)
            self.D_spe.weight_init(mean=0.0, std=0.02)
            # weight initial
            self.D_spa.to(self.device)
            self.D_spe.to(self.device)

    def set_optimizers(self):
        # Adam optimizer
        betas_set = (self.opt.beta1, self.opt.beta2)
        self.G_optimizer = optim.Adam([{'params': self.Abunreger.parameters(), 'initial_lr': self.opt.lrG}], lr=self.opt.lrG,
                                 betas=betas_set)
        self.R_optimizer = optim.Adam([{'params': self.Reslearner.parameters(), 'initial_lr': self.opt.lrR}], lr=self.opt.lrR,
                                 betas=betas_set)
        self.D_spa_optimizer = optim.Adam([{'params': self.D_spa.parameters(), 'initial_lr': self.opt.lrD}], lr=self.opt.lrD,
                                     betas=betas_set)
        self.D_spe_optimizer = optim.Adam([{'params': self.D_spe.parameters(), 'initial_lr': self.opt.lrD}], lr=self.opt.lrD,
                                     betas=betas_set)

        self.G_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.G_optimizer, T_max=self.opt.G_Decay_epoch,
                                                                 eta_min=0.0000001, last_epoch=-1)
        self.R_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.R_optimizer, T_max=self.opt.G_Decay_epoch,
                                                                 eta_min=0.0000001, last_epoch=-1)
        self.D_spa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.D_spa_optimizer, T_max=self.opt.D_Decay_epoch,
                                                                     eta_min=0.0000001, last_epoch=-1)
        self.D_spe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.D_spe_optimizer, T_max=self.opt.D_Decay_epoch,
                                                                     eta_min=0.0000001, last_epoch=-1)

    def load_checkpoint(self,load_epoch='0'):
        if self.mode is 'Train' and load_epoch == '0':
            load_epoch = str(self.opt.start_epoch)
        # load existing models
        G_dir = os.path.join(self.model_root, 'generator_param_' + load_epoch + '.pth')
        R_dir = os.path.join(self.model_root,'reslearner_param_' + load_epoch + '.pth')
        if self.mode is 'Train': # training, load epoch or from epoch 0
            if self.opt.start_epoch <= self.opt.D_start_epoch:
                if os.path.exists(G_dir):
                    self.Abunreger.load_state_dict(torch.load(G_dir))
                    self.start_epoch = self.opt.start_epoch
                    print('load successful!')
                else:
                    self.start_epoch = 0
                    print('load failed, training from initial!')
            else:
                D_spa_dir = os.path.join(self.model_root, 'discriminator_spa_param_' + load_epoch + '.pth')
                D_spe_dir = os.path.join(self.model_root, 'discriminator_spe_param_' + load_epoch + '.pth')
                if os.path.exists(G_dir) and os.path.exists(D_spa_dir) and os.path.exists(D_spe_dir):
                    self.Abunreger.load_state_dict(torch.load(G_dir))
                    self.D_spa.load_state_dict(torch.load(D_spa_dir))
                    self.D_spe.load_state_dict(torch.load(D_spe_dir))
                    self.start_epoch = self.opt.start_epoch
                    print('load successful!')
                else:
                    self.start_epoch = 0
                    print('load failed, training from initial!')
        else: # testing or evaluating, check and load models
            if os.path.exists(G_dir):
                self.Abunreger.load_state_dict(torch.load(G_dir))
                print('load G successful!')
            else:
                print('model G not exist,please check!')
                exit()
        if os.path.exists(R_dir):
            self.Reslearner.load_state_dict(torch.load(R_dir))
            print('load R successful!')
        else:
            if self.mode is 'Test':
                print('model R not exist,please check!')
                exit()

    def set_logger(self,train_loader,test_loader):
        self.Train_logger = Logger(self.opt.train_epoch, len(train_loader), window_name=self.opt.identy_root)
        self.Test_logger = Logger(self.opt.train_epoch, len(test_loader), window_name=self.opt.identy_root)

    def training_loop(self,train_loader,test_loader):
        print('training start!')
        psnr_best = 0 #
        for epoch in range(self.start_epoch + 1, self.opt.train_epoch):
            # train one epoch
            self.train_one_epoch(train_loader,epoch)
            # update scheduler
            if epoch > self.opt.lrd_start_epoch:
                self.G_scheduler.step()
                if (epoch > self.opt.D_start_epoch):
                    self.D_spe_scheduler.step()
                    self.D_spa_scheduler.step()
            # eval
            self.eval_one_epoch(test_loader, epoch)
            if self.eval_psnr > psnr_best and (epoch > self.opt.save_epoch): # save best models
                self.save_checkpoints(self.opt.train_epoch, self.model_root)
                psnr_best = self.eval_psnr
            if (epoch % self.opt.save_epoch == 0):
                self.save_checkpoints(epoch, self.model_root)

    def save_log(self):
        self.Train_logger.save(os.path.join(self.root,self.opt.identy_root + '.log'), self.opt.identy_root)

    def train_one_epoch(self,train_loader,epoch):
        num_iter = 0
        for x_, y_ in train_loader:
            x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
            # generation
            self.Abunreger.zero_grad()  #
            self.Reslearner.zero_grad()
            G_result_LMM,abundance,brightness,G_result_Res = self.generation_forward(x_,y_) #linear mixture,xxx,xx,other mixture
            G_result = G_result_LMM + G_result_Res
            self.loss_functions.calcul_generation_loss_iter(y_,G_result_Res,G_result,abundance)
            # discrimination
            if epoch > self.opt.D_start_epoch:
                # train discriminator D
                self.D_spa.zero_grad()
                self.D_spe.zero_grad()
                D_real_result = self.D_spa(x_, y_[:,self.close_list,:,:]).squeeze()
                spe_pre_real, spe_pre_result = self.D_spe(y_[:,self.close_list,:,:], G_result[:,self.close_list,:,:])
                D_pre_result = self.D_spa(x_, G_result[:,self.close_list,:,:]).squeeze()
                self.loss_functions.calcul_D_loss_iter(D_real_result,D_pre_result,spe_pre_real,spe_pre_result)
                # optimize D
                if (num_iter % self.opt.G_iter == 0):
                    self.loss_functions.loss['D_loss'].backward(retain_graph=True)  #
                    self.D_spa_optimizer.step()  #
                    self.D_spe_optimizer.step()  #
                    spe_pre_real, spe_pre_result = self.D_spe(y_[:,self.close_list,:,:], G_result[:,self.close_list,:,:])
                    D_pre_result = self.D_spa(x_, G_result[:,self.close_list,:,:]).squeeze()
                self.loss_functions.calcul_G_adver_loss_iter(D_pre_result,spe_pre_result)

            self.loss_functions.loss['G_train_loss'].backward()
            # nn.utils.clip_grad_norm_(G.parameters(),max_norm=1)
            self.G_optimizer.step()
            self.R_optimizer.step()

            num_iter += 1
            self.Train_logger.log(self.loss_functions.loss,images={'MSI': x_, 'real_HSI': y_, 'fake_HSI': G_result,'LMM_HSI':G_result_LMM,'Res_HSI':G_result_Res}) #

    def eval_one_epoch(self,test_loader,epoch):
        self.Reslearner.eval()
        if self.mode is 'Test':
            if self.cal_performance:
                f = open(self.save_root + '/performance.txt', 'a')
            print('test start!')
        else:
            batch_psnr = 0
        for i, data in enumerate(test_loader):
            x_, y_ = data
            y_, x_ = y_.cuda(), x_.cuda()
            with torch.no_grad():
                G_result_LMM,abundance,brightness,G_result_Res = self.generation_forward(x_, y_)#_LMM, G_result_Res
            G_result = G_result_LMM + G_result_Res
            if self.mode is 'Train':
                y_ = y_[:,self.close_list,:,:]
                G_result = G_result[:,self.close_list,:,:]
                psnr = batch_PSNR(y_ / 16, G_result / 16, 65535)
                batch_psnr += psnr
                self.Test_logger.log({'psnr_test': psnr}, images={'MSI': x_, 'real_HSI': y_, 'fake_HSI': G_result,'LMM_HSI':G_result_LMM,'Res_HSI':G_result_Res})#
            else:
                s = test_loader.dataset.hyper_path[i][0:-4]
                if self.abun_flag:
                    abun = abundance[0].cpu().detach().numpy()
                    bright = brightness[0].cpu().detach().squeeze().numpy()
                    scio.savemat(os.path.join(self.save_root, s + '_abundance.mat'), {'abundance': abun, 'bright': bright})

                hyper_image = (G_result[0].cpu().detach().numpy() * 4095).astype('uint16')  # chw
                path = os.path.join(self.save_root, s + '.tif')
                write_img(path, hyper_image)
                if self.cal_performance:
                    hyper = (y_[0].cpu().detach().numpy() * 4095).astype('float64')
                    rmse,mrae,sam,mssim,mpsnr = self.cal_performance.calcul_one_img(hyper_image, hyper)
                    f.write('\n')
                    f.write(s)
                    f.write('    rmse=' + str(rmse) + '   mrae=' + str(mrae) + '   sam=' + str(sam) + '   ssim=' + str(
                        mssim) + '   psnr=' + str(mpsnr))
                    f.write('\n')
                print('%d images generation complete!' % (i+1))
        if self.mode is 'Test':
            if self.cal_performance:
                mean_rmse,mean_mrae,mean_sam,mean_ssim,mean_psnr = self.cal_performance.calcul_mean()

                f.write('mean of all images:\n')
                f.write('rmse   '+ str(mean_rmse)+'\n' + 'mrae   ' + str(mean_mrae)+'\n' + 'sam   ' + str(
                    mean_sam)+'\n' + 'ssim   ' + str(mean_ssim)+'\n' + 'psnr   ' + str(mean_psnr))
                f.write('\n')
                f.close()
        else:
           self.eval_psnr = batch_psnr/len(test_loader)
           self.Reslearner.train()
    def generation_forward(self,x_,y_):
        pos, brightness, feature = self.Abunreger(x_)
        abundance = F.softmax(pos,dim=1)
        G_result_LMM = brightness * torch.matmul(self.spe_lib, torch.reshape(abundance.permute(1, 0, 2, 3),
                                                                             (abundance.shape[1], -1))).reshape(
            y_.shape[1],
            y_.shape[0],
            y_.shape[2],
            y_.shape[3]).permute(1, 0, 2, 3)
        G_result_Res = self.Reslearner(feature,G_result_LMM,self.spe_lib)  # detach?,abundance
        return G_result_LMM,abundance,brightness,G_result_Res

    def save_checkpoints(self,epoch,model_root):
        if epoch > self.opt.D_start_epoch:
            torch.save(self.D_spa.state_dict(), model_root + '/discriminator_spa_param_' + str(epoch) + '.pth')
            torch.save(self.D_spe.state_dict(), model_root + '/discriminator_spe_param_' + str(epoch) + '.pth')

        torch.save(self.Abunreger.state_dict(), model_root + '/generator_param_' + str(epoch) + '.pth')
        torch.save(self.Reslearner.state_dict(), model_root + '/reslearner_param_' + str(epoch) + '.pth')

