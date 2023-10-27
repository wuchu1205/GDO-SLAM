
cfg = dict(
    model_type='gdo_seg',
    n_cats=2,
    num_aux_heads=0,
    lr_start=1e-3,
    weight_decay=5e-4,
    warmup_iters=500,
    max_iter=100000,
    dataset='CityScapes',
    im_root='/home/dzt-uav/wc/dataset/cityscapes',
    train_im_anns='/home/dzt-uav/wc/dataset/cityscapes/train.txt',
    val_im_anns='/home/dzt-uav/wc/dataset/cityscapes/val.txt',
    scales=[0.75, 2.],
    cropsize=[1024, 1024],
    eval_crop=[1024, 1024],
    eval_scales=[1.0],
    ims_per_gpu=16,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
    finetune_from="/home/dzt-uav/wc/vision_segmentation/BiSeNet-master/tools/res_used_for_ral_paper/model_best.pth"
)
