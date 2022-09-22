import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
from torch.nn import init
from torch.nn.utils.spectral_norm import spectral_norm


class Reslearner(nn.Module):
    def __init__(self,opt,output_nc = 224):
        super(Reslearner,self).__init__()
        # self.opt = opt
        self.feature_nc = 2**(opt.downscale_factor-1) * opt.ngf #input feature dims
        self.downsample_factor = 2**(opt.downscale_factor) #down scale factor
        self.input_size = opt.crop_size // self.downsample_factor  # size of the feature
        self.input_nc = opt.abun_num #345
        self.output_nc = output_nc # output hsi channel 224
        self.nf = opt.ngf//2 # feature_dim for MLP now 64
        self.mlp_depth = opt.mlp_depth # mlp depth
        self.position_encode = opt.position_encode
        # if self.training:
        #     self.img_size = self.crop_size
        # else:
        #     self.img_size = 256

        self.res_generator = ResGenerator(downscale_factor=opt.downscale_factor,input_nc=self.input_nc,output_nc=self.output_nc,
                                          width=self.nf,depth=self.mlp_depth,position_encode=self.position_encode,device = 'cuda:0')
        self.num_params = self.res_generator.num_params
        self.param_seter = ParamSeter(self.feature_nc,self.num_params,d = self.nf) #input_nc,out_nc,d = 64


    def weight_init(self, init_type='normal', gain=0.002):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            '''

            '''

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
    def forward(self,input,abun_feature,spe_lib):
        param_dis = self.param_seter(input)
        output = self.res_generator(param_dis,abun_feature,spe_lib)
        return output

class ResGenerator(nn.Module):    
    def __init__(self,downscale_factor = 2, input_nc = 345, output_nc = 224, width = 64, depth = 5, position_encode = 'cosine',device = 'cuda:0'):
        # pass
        super(ResGenerator, self).__init__()
        self.downsampling_factor = downscale_factor
        self.lib_num = input_nc        # 345
        self.spe_channel = output_nc     # 224
        self.position_encode = position_encode
        self.xy_coords = None
        self.width = width
        self.depth = depth
        self.channels = []
        self._set_channels()
        self.num_params = 0
        self.splits={}
        self._set_num_params()
        self.device = device

    def _set_channels(self):
        input_channel = 0 #self.spe_channel  #224
        if self.position_encode =='cosine':
            input_channel +=(4*self.downsampling_factor)
        self.channels = [input_channel]
        for _ in range(self.depth-1):
            self.channels.append(self.width)
            # self.channels.append(self.lib_num)
            # self.channels.append(self.spe_channel)
        #outlayer
        self.channels.append(self.lib_num)
    def _set_num_params(self):
        nparams = 0
        self.splits={
            "biases":[],
            "weights":[],
        }
        # the position of current parameter in the parameter queue
        idx=0
        for layer,nc_in in enumerate(self.channels[:-1]):# position of the parameter for each layer
            nc_out=self.channels[layer+1]
            nparams +=nc_out # the offset
            self.splits["biases"].append((idx,idx+nc_out))
            idx+=nc_out
            nparams += nc_in*nc_out
            self.splits["weights"].append((idx,idx+nc_out*nc_in))
            idx += nc_out*nc_in

        self.num_params = nparams * 2
    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]


    def forward(self,lr_params,hyper_linear,spe_lib):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"
        k = int(2 ** self.downsampling_factor)  #  down scale factor
        bs, _, h_lr, w_lr = lr_params.shape
        h, w = h_lr * k, w_lr * k

        if not (self.position_encode is None):
            if self.xy_coords is None or (self.xy_coords.shape[2] != h) or (self.xy_coords.shape[3] != w):
                self.xy_coords = _get_coords(bs, h, w, self.device, float(k), self.position_encode)
        # input_code = torch.cat([hyper_linear, self.xy_coords], 1)
        input_code = self.xy_coords
        nc_input = input_code.shape[1]  # input channels the encoded position
        tiles = input_code.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(bs, h_lr, w_lr, int(k * k), nc_input)
        out = tiles
        num_layers = len(self.channels) - 1
        for idx, nc_in in enumerate(self.channels[:-1]):
            nc_out = self.channels[idx + 1]

            # catch parameter from lr_params
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]
            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nc_in, nc_out)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nc_out)
            out = torch.matmul(out, w_) + b_
            if idx < num_layers - 1:
                out = F.leaky_relu(out, 0.01, inplace=True)
            else:
                out = 0.1 * torch.tanh(out)  # 0.01
        out = torch.matmul(out, spe_lib.permute(1, 0))  # the weight of spectral lib
        out = out.view(bs, h_lr, w_lr, k, k, self.spe_channel).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.spe_channel, h, w)
        ref2 = torch.matmul(out, hyper_linear)
        out = tiles
        offset = lr_params.shape[1] // 2
        for idx, nc_in in enumerate(self.channels[:-1]):
            nc_out = self.channels[idx + 1]
            # catch parameter from lr_params
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)
            w_ = lr_params[:, wstart+offset:wstop+offset]
            b_ = lr_params[:, bstart+offset:bstop+offset]
            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nc_in, nc_out)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nc_out)
            out = torch.matmul(out, w_) + b_
            if idx < num_layers - 1:
                out = F.leaky_relu(out, 0.01, inplace=True)
            else:
                out = 0.1 * torch.tanh(out)  # 0.01
        out = torch.matmul(out, spe_lib.permute(1, 0))  # the weight of spectral lib
        out = out.view(bs, h_lr, w_lr, k, k, self.spe_channel).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.spe_channel, h, w)
        ref3 = torch.matmul(out, ref2)

        out = ref2 + ref3

        return out

def _get_coords(bs, h, w, device, ds, coords_type):
    """Creates the position encoding for the pixel-wise MLPs"""
    if coords_type == 'cosine':
        f0 = ds
        f = f0
        while f > 1:
            x = torch.arange(0, w).float()
            y = torch.arange(0, h).float()
            xcos = torch.cos((2 * pi * torch.remainder(x, f).float() / f).float())
            xsin = torch.sin((2 * pi * torch.remainder(x, f).float() / f).float())
            ycos = torch.cos((2 * pi * torch.remainder(y, f).float() / f).float())
            ysin = torch.sin((2 * pi * torch.remainder(y, f).float() / f).float())
            xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            coords_cur = torch.cat([xcos, xsin, ycos, ysin], 1).to(device)
            if f < f0:
                coords = torch.cat([coords, coords_cur], 1).to(device)
            else:
                coords = coords_cur
            f = f//2
    else:
        raise NotImplementedError()
    return coords.to(device)

class ParamSeter(nn.Module):
    def __init__(self,input_nc,out_nc,d = 64):
        super(ParamSeter, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(input_nc,d,3,1,0)),# supernetwork
            # nn.InstanceNorm2d(d),
            nn.ReLU()
        )
        # self.conv2 = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(d, d * 2, 3, 1, 0),
        #     nn.InstanceNorm2d(d*2),
        #     nn.ReLU()
        # )
        self.conv3 = nn.Sequential(
            nn.Conv2d(d,out_nc,1,1,0),
            nn.ReLU()
        )
    def forward(self,input_feature):
        x=self.conv1(input_feature)
        # x=self.conv2(x)
        x=self.conv3(x)
        return x
