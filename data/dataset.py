# coding:utf8
import os
from torch.utils import data
import numpy as np
import scipy.io as scio
from . import xlsx_operation


class ParticleTrajectory(data.Dataset):

    def __init__(self, root, per, train=True, test=False,ratio=0.7, k=0):
        """
        get the tracking result, divide the train/validation/test dataset
        """

#        import ipdb
#        ipdb.set_trace()

        self.test = test
        data = [os.path.join(root, sample) for sample in os.listdir(root)]
        data_num = len(data)
        
        # sort the file
        data = sorted(data, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        
        # shuffle the data except test data
        if len(per) ==1:
                print('test: not shuffle')
        else:
            for i in range(len(per)):
                data[i] = data[per[i]]
        

        # test
        if self.test:
            self.data = data
        # train
        elif train:
            self.data = data[:int(ratio * data_num)]
            # k-flod validation
            #self.data = data[:int((k%10)*data_num/10)] + data[int((k%10+1)*data_num/10):]
        # validation
        else:
            self.data = data[int(ratio * data_num):]
            # k-flod validation
            #self.data = data[int((k%10)*data_num/10):int((k%10+1)*data_num/10)]
            #print('index={}-{}'.format(int((k%10)*data_num/10),int((k%10+1)*data_num/10)))

        

    def __getitem__(self, index):
        """
        return a sample
        """
        data_path = self.data[index]
        data = xlsx_operation.read_excel_xlsx(data_path,'Sheet1')
        feature_dim = data.shape[1]-1        

        # read data and label from feature extraction xlsx
        sample = data[:,0:feature_dim]
        label = data[:,feature_dim].reshape(len(data),1)
        return sample, label

    def __len__(self):
        """
        return the length of samples
        """
        return len(self.data)




