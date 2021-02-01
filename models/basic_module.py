#coding:utf8
import torch as t
import time


class BasicModule(t.nn.Module):
    """
    pack nn.Module and provide save and load method
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self)) # default name

    def load(self, path):
        """
        load model from path
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        save model, name=model name + time
        """
        if name is None:
            #prefix = 'checkpoints/' + self.model_name + '_'
            #name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            prefix = 'checkpoints/' + self.model_name.split('.')[-1].split('\'')[-2]
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


