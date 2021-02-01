# coding:utf8
import warnings
import torch as t
import numpy as np

class DefaultConfig(object):
    
    #print('*************visdom config*****************')
    env = 'default'  # visdom environment
    vis_port =8097 # visdom port

    #print('*************path config*****************')
    train_data_root = './data/data_20200304/train_process/50/'  # train data path
    test_data_root = './data/data_20200304/test_origin/'  # test data path
    test_result_root = './data/data_20200304/test/' # test result path
    load_model_path = None  # load pretrained model path

    #print('*************debug config*****************')
    print_freq = 500  # print info every N batch
    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb

    #print('*************train config*****************')
    ksplit = 1 # repeat train for ksplit times
    trainset_ratio = 0.75
    batch_size = 256  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 32  # how many workers for loading data
    max_epoch = 30 # epoch
    lr = 1e-3  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # weight decay

    #print('*************model config*****************')
    in_dim = 2 # feature dim
    hidden_dim = 128 # hidden dim of lstm
    n_layer = 1 # layer of lstm
    n_class = 3 # num of class
    use_attention = False # use attention layer or not
    #attention_win = np.array([1,2,3,4,5]) # attention window size
    #attention_win = [1,2,3,4,5]
    attention_win = 1 # attention window size
    
    #print('*************criterion config*****************')
    use_customloss = False
    delta = np.array([5,10,15])
    alpha = np.array([0.9,0.8,0.7,0.6,0.5])
    beta = np.array([0.1,0.2,0.3,0.4,0.5])

    #print('*************metrics config*****************')
    iou_threshold = np.array([0.75,0.875,1])



    def _parse(self, kwargs):
        """
        update config based on kwargs dictionary
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
