#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import json
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg



## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='../configs/groundsegv2.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)


def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        msg = net.load_state_dict(torch.load(args.finetune_from,
            map_location='cpu'), strict=False)
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7, lb_ignore)
    print('ok')
    return net, criteria_pre


def set_optimizer(model):
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if param.dim() == 1:
            non_wd_params.append(param)
        elif param.dim() == 2 or param.dim() == 4:
            wd_params.append(param)
    params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_model_dist(net):
    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        #  find_unused_parameters=True,
        output_device=local_rank
        )
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')

    return time_meter, loss_meter, loss_pre_meter



def train():
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader(cfg, mode='train')

    ## model
    net, criteria_pre = set_model(dl.dataset.lb_ignore)

    ## optimizer
    optim = set_optimizer(net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    # net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    best_miou = 0.0
    torch.cuda.empty_cache()
    ## train loop
    for epoch in range(0,540):
        net.train( )
        for it, (im, lb) in enumerate(dl):
            im = im.cuda()
            lb = lb.cuda()

            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            with amp.autocast(enabled=cfg.use_fp16):
                logits = net(im)
                loss_pre = criteria_pre(logits, lb)

                loss = loss_pre
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())

            ## print training log message
            if (it + 1) % 60 == 0:
                lr = lr_schdr.get_lr()
                lr = sum(lr) / len(lr)
                print_log_msg(
                    epoch*180+it, cfg.max_iter, lr, time_meter, loss_meter,
                    loss_pre_meter)
            lr_schdr.step()

        ## dump the final model and evaluate the result
        save_pth = osp.join(cfg.respth, 'model_final.pth')
        save_best_pth = osp.join(cfg.respth, 'model_best.pth')
        logger.info('\nsave models to {}'.format(save_pth))
        state = net.state_dict()
        torch.save(state, save_pth)
        if epoch % 4 == 0:
            logger.info('\nevaluating the final model')
            #torch.cuda.empty_cache()
            iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net)

            logger.info('\neval results of f1 score metric:')
            logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
            logger.info('\neval results of miou metric:')
            logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))
            miou = iou_content[3][2]
            if best_miou < float(miou):
                best_miou = float(miou)
                torch.save(state, save_best_pth)
                print("miou is: ",best_miou)
                logger.info('\nsave models to {}'.format(save_best_pth))



    return


def main():
    torch.cuda.set_device(0)
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train()


if __name__ == "__main__":
    main()
