import os
import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='',)
parse.add_argument('--weight-path', type=str, default='',)
parse.add_argument('--img-path', dest='img_path', type=str, default='',)
parse.add_argument('--label-path', dest='label_path', type=str, default='',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='train')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
img_root = args.img_path
label_root = args.label_path
label_names = [label_path for label_path in os.listdir(label_root)]
img_names = [img_path for img_path in os.listdir(img_root)]
for img_name in img_names:
    img_path = os.path.join(img_root,img_name)
    im = cv2.imread(img_path)[:, :, ::-1]
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
    out = net(im)
    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
    out = out.argmax(dim=1)
    out[out==1]=0

    # visualize
    out = out.squeeze().detach().cpu().numpy()
    pred = out * 255
    # pred = palette[out]
    save_dir = ""
    save_path = os.path.join(save_dir,img_name+'.jpg')
    cv2.imwrite(save_path, pred)