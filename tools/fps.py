import sys
import time

sys.path.insert(0, '.')
import torch
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from lib.models import model_factory
import argparse
from configs import set_cfg_from_file

import torch
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='../configs/groundseg.py',)
    return parse.parse_args()



args = parse_args()
cfg = set_cfg_from_file(args.config)
model = model_factory[cfg.model_type](cfg.n_cats)
device = torch.device('cuda')
# compute_speed(net, (1, 3, 1024, 2048), 0)
model.eval()
model.to(device)
iterations = None
input = torch.randn(1, 3, 1024, 2048).cuda()
with torch.no_grad():
    for _ in range(10):
        model(input)

    if iterations is None:
        elapsed_time = 0
        iterations = 100
        while elapsed_time < 1:
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(iterations):
                model(input)
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            iterations *= 2
        FPS = iterations / elapsed_time
        iterations = int(FPS * 6)

    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start
    latency = elapsed_time / iterations * 1000
torch.cuda.empty_cache()
FPS = 1000 / latency
print(FPS)