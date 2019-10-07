from .base_options import BaseOptions

class TransformerOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--name_landmarks_list', type=str, help='name of landmarks list for A and B')
        self.parser.add_argument('--which_target', type=int, default=0, help='which target')
        self.parser.add_argument('--default_r', type=int, default=1, help='default r')
        self.parser.add_argument('--fineSize_F1', type=int, default=256, help='fine size of F1 for network')
        self.parser.add_argument('--pca_dim', type=int, default=3)
        self.parser.add_argument('--lam_align', type=float, default=10)
        self.parser.add_argument('--lam_pix', type=float, default=10)
        self.parser.add_argument('--bound_size', type=int, default=64)

        self.parser.add_argument('--max_step', type=int, default=200000)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--weight_decay', type=float, default=0.0001)

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_path', type=str, default='')
        self.parser.add_argument('--which_iter', type=str, default='0')
        self.parser.add_argument('--random_seed', type=int, default=123)

        self.isTrain = True
