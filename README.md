# GDO-SLAM
The source code of ground segmentation module of GDO-SLAM. The code for SLAM will be released in the future.

## eval the accuracy of pretrained models

Before run the code, you need to change the field of `im_root` and `train/val_im_anns` in the config file. Then you can evaluate a trained model like this:
```
$ python tools/eval.py --config /config/path --weight-path /weight/path
```

To test the fps, you can run:
```
$ python tools/fps.py --config /config/path
```
To inference single images, you can run:
```
$ python tools/demo.py --config /config/path --weight-path /weight/path --img-path /image/path
```
