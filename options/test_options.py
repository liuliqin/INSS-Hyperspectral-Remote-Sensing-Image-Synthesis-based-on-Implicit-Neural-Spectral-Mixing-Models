from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def __init__(self):
        self.initialized = False

    def initialize(self,parser):
        BaseOptions.initialize(self,parser)
        parser.add_argument('--crop_size', type=int, default=256, help='input size')
        parser.add_argument('--model_epoch', default='201', help='Which model to test?')  # 180,40
        parser.add_argument('--test_flag', default='test_mini', help='train or test_mini')  # _mini
        parser.add_argument('--abun_flag', default=True, help='save abundance mat')
        parser.add_argument('--cal_performance', default=False, help='cal performance?')

        self.initialized =True
        return parser