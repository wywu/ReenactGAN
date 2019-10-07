from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--model_dir', type=str, default='./pretrained_models', help='pretrained models are saved here')
        self.parser.add_argument('--name_list', type=str, help='name of landmarks list for test images')
        self.parser.add_argument('--save_root_path', type=str, help='root path of save images splitly.')
        self.parser.add_argument('--real_F1_path', type=str, help='path of save images of real F1.')
        self.parser.add_argument('--Boundary_path', type=str, help='path of save images of Boundary.')
        self.parser.add_argument('--Boundary_transformed_path', default='', type=str, help='path of save images of transformed Boundary.')
        self.parser.add_argument('--fake_F2_path', type=str, help='path of save images of fake F2.')
        self.parser.add_argument('--fineSize_F1', type=int, default=256, help='fine size of F1 for network')
        self.parser.add_argument('--fineSize_Boundary', type=int, default=64, help='fine size of Boundary for network')
        self.parser.add_argument('--nc_F1', type=int, default=3, help='# of channels of F1')
        self.parser.add_argument('--nc_F2', type=int, default=3, help='# of channels of F2')
        self.parser.add_argument('--nc_Boundary', type=int, default=15, help='# of channels of Boundary')
        self.parser.add_argument('--num_stacks', type=int, default=2, help='num of stacks of hourglass network')
        self.parser.add_argument('--num_blocks', type=int, default=1, help='uumber of residual modules at each location in the hourglass')
        self.parser.add_argument('--which_boundary_detection', type=str, default='v0', help='which version of boundary detection model')
        self.parser.add_argument('--which_decoder', type=str, default='trump', help='which version of decoder model')
        self.parser.add_argument('--which_transformer', type=str, default='trump', help='which version of transformer model')

        self.isTrain = False
