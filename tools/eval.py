from evaluate import groundseg_eval_model
from lib.models import model_factory
import argparse
from configs import set_cfg_from_file
from lib.data import get_data_loader
import logging
import torch
from tabulate import tabulate
# net = torch.hub.load()
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='')
    parse.add_argument('--weight-path', type=str, default=None)
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)
dl = get_data_loader(cfg, mode='train')
net = model_factory[cfg.model_type](cfg.n_cats)
net.load_state_dict(torch.load(args.weight_path,
                                     map_location='cpu'), strict=True)
net.cuda()

iou_heads, iou_content, f1_heads, f1_content = groundseg_eval_model(cfg, net)
logger = logging.getLogger()
logger.info('\neval results of miou metric:')
print('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))
