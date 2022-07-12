from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=300, help='how many test images to run')        
        self.parser.add_argument('--use_real_img', action='store_true', help='use real image for first frame')
        self.parser.add_argument('--start_frame', type=int, default=0, help='frame index to start inference on')    
        self.parser.add_argument('--n_frames_total', type=int, default=30, help='frame index to start inference on') 
        self.parser.add_argument('--max_frames_per_gpu', type=int, default=1, help='max number of frames to load into one GPU at a time')
        self.parser.add_argument('--max_frames_backpropagate', type=int, default=1, help='max number of frames to backpropagate') 
        self.parser.add_argument('--subset', action='store_true', help='if is a subset for face')
        self.parser.add_argument('--max_t_step', type=int, default=1, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')
        
        self.parser.add_argument('--load_sequence', action='store_true', help='if load a sequence')    

        
        self.isTrain = False
