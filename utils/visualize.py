# coding:utf8
import visdom
import time
import numpy as np


class Visualizer(object):
    """
    pack visdom, you can still use the original visdom inference by `self.vis.function`
    
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env,use_incoming_socket=False, **kwargs)
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        modify visdom config
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot many data 
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    

    def plot(self, name,xname,yname,x, y, **kwargs):
        """
        plot a point 
        self.plot('loss',1.00)
        """
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name,xlabel=xname,ylabel=yname),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )


    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
