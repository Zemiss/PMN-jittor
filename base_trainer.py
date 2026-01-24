import random
import argparse
import jittor as jt

from utils import *
from archs import *
from losses import *
from pathlib import Path

class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/Ours.yml", type=Path, help="path to config")
        self.parser.add_argument('--mode', '-m', default=None, type=str, help="train or test")
        self.parser.add_argument('--save_plot', '-s', default=True, type=bool, help="save or not")
        self.parser.add_argument('--debug', '-d', default=False, type=bool, help="debug or not")

        return self.parser.parse_args()

class Base_Trainer():
    def __init__(self):
        self.initialization()
    
    def get_lr_lambda_func(self):
        num_of_epochs = self.hyper['stop_epoch'] - self.hyper['last_epoch']
        step_size = self.hyper['step_size']
        T = self.hyper['T'] if 'T' in self.hyper else 1 
        if 'cos' in self.hyper['lr_scheduler'].lower():
            self.lr_lambda = lambda x: get_cos_lr(x, period=num_of_epochs//T, lr=self.hyper['learning_rate'], peak=step_size)
        elif 'multi' in self.hyper['lr_scheduler'].lower():
            self.lr_lambda = lambda x: get_multistep_lr(x, period=num_of_epochs//T, decay_base=1,
                                        milestone=[step_size, step_size*9//5], gamma=[0.5, 0.1], 
                                        lr=self.hyper['learning_rate'])
        return self.lr_lambda

    # 不这么搞随机jittor和numpy的联动会出bug，随机种子有问题
    def worker_init_fn(self, worker_id):
        jt_seed = random.randint(0, 2**31)
        random.seed(jt_seed + worker_id)
        if jt_seed >= 2**30:  # make sure jt_seed + workder_id < 2**32
            jt_seed = jt_seed % 2**30
        np.random.seed(jt_seed + worker_id)

    def initialization(self):
        parser = BaseParser()
        self.parser = parser.parse()
        with open(self.parser.runfile, 'r', encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.mode = self.args['mode'] if self.parser.mode is None else self.parser.mode
        self.save_plot = self.parser.save_plot
        if self.parser.debug:
            self.args['num_workers'] = 0
        if 'clip' not in self.args['dst']: 
            self.args['dst']['clip'] = False
        self.dst = self.args['dst']
        self.hyper = self.args['hyper']
        self.arch = self.args['arch']
        # Jittor 自动管理设备，通过flags配置
        jt.flags.use_cuda = 1 if jt.has_cuda else 0
        self.device = 'cuda' if jt.has_cuda else 'cpu'
        self.hostname = socket.gethostname()
        self.model_name = self.args['model_name']
        self.model_dir = self.args['checkpoint']
        self.sample_dir = os.path.join(self.args['result_dir'] ,f"samples-{self.model_name}")
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.sample_dir+'/temp', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./checkpoints', exist_ok=True)
        os.makedirs('./metrics', exist_ok=True)

class LambdaScheduler:
    """学习率调度器 - Jittor版本"""
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda if isinstance(lr_lambda, list) else [lr_lambda]
        self.last_epoch = last_epoch
        # Jittor优化器使用lr属性而不是param_groups
        self.base_lr = optimizer.lr
        self._last_lr = [optimizer.lr]
        
    def get_lr(self):
        # lr_lambda函数返回的是绝对学习率，不是乘数
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambda]
    
    def get_last_lr(self):
        return self._last_lr
    
    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        values = self.get_lr()
        self._last_lr = values
        
        # Jittor优化器直接设置lr属性
        if len(values) > 0:
            self.optimizer.lr = values[0]

def get_cos_lr(step, period=1000, peak=20, lr=1e-4, ratio=0.2):
    T = step // period
    decay = 2 ** T
    step = step % period
    if step <= peak and T>0:
        mul = step / peak
    else:
        mul = (1-ratio) * (np.cos((step - peak) / (period - peak) * math.pi) * 0.5 + 0.5) + ratio
    return lr * mul / decay

def get_multistep_lr(step, period=1000, lr=1e-4, milestone=[500, 900], gamma=[0.5, 0.1], decay_base=1):
    decay = decay_base ** (step // period)
    step = step % period
    mul = 1
    for i in range(len(milestone), 0, -1):
        if step > milestone[i-1]:
            mul = gamma[i-1]
            break
    return lr * mul / decay
