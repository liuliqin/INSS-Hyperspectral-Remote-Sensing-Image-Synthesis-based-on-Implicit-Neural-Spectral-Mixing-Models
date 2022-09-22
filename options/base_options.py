import argparse,os

class BaseOptions():
    def __init__(self):
        self.initialized = False
    def initialize(self,parser):
        parser.add_argument('--dataset', required=False, default='mini_aviris_2202', help='')#dataset
        parser.add_argument('--device', required=False, default='cuda:0', help='device GPU ID')
        parser.add_argument('--input_channel', type=int, default=3, help='input msi channel')
        parser.add_argument('--lib_path', required=False, default='spe_chose_345.mat', help='spectral lib path')
        parser.add_argument('--abun_num', type=int, default=345, help='spectral lib path')
        parser.add_argument('--ngf', type=int, default=128)
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--downscale_factor', type=int, default=4, help='down scale factor 2 ** downscale_factor')
        parser.add_argument('--mlp_depth', type=int, default=2, help='MLP depth for residual learn')
        parser.add_argument('--position_encode', required=False, default='cosine', help='MLP depth for residual learn')
        parser.add_argument('--res_decay', type=float, default=0.1, help='decay coefficient of high orders')
        parser.add_argument('--save_root', required=False, default='results', help='results save path')
        parser.add_argument('--identy_root', required=False, default='20220629 order3', help='identy path')#
        self.initialized = True
        return parser
    def parse(self,save = True):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        if save:
            root = os.path.join(opt.save_root, opt.dataset, opt.identy_root)  #
            if not os.path.isdir(root):
                os.mkdir(root)
            file_name = os.path.join(root, 'options.txt')
            with open(file_name, 'w') as f:
                for key, value in opt.__dict__.items():
                    f.write(key + '\t' + str(value) + '\n')
        self.opt = opt
        return self.opt
