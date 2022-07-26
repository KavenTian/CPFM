config=yolox_csp_llvip
epoch=9
# python tools/misc/vis_feat.py dataset/KAIST/coco_kaist/val/val_visible.txt dataset/KAIST/coco_kaist/val/val_lwir.txt configs/multispec/${config}.py work_dirs/${config}/epoch_${epoch}.pth
python tools/misc/vis_feat.py /dataset/llvip_coco/val/images_vis.txt /dataset/llvip_coco/val/images_inf.txt configs/multispec/${config}.py work_dirs/${config}/epoch_${epoch}.pth --score-thr=0.5