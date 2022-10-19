# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import numpy as np
import cv2

from mmdet.apis import (async_inference_detector, inference, inference_rgbt_detector,
                        init_detector, show_result_pyplot, show_rgbt_result_pyplot)
import mmcv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def forward_vis_hook(module, data_input, data_output):
    fmap_vis.append(data_output)
    input_vis.append(data_input)

def backward_vis_hook(module, grad_in, grad_out):
    grad_vis.append(grad_out[0].detach())

def forward_inf_hook(module, data_input, data_output):
    fmap_inf.append(data_output)
    input_inf.append(data_input)

def backward_inf_hook(module, grad_in, grad_out):
    grad_inf.append(grad_out[0].detach())

# def forward_pub_hook(module, data_input, data_output):
#     fmap_pub.append(data_output)
#     input_pub.append(data_input)

# def backward_pub_hook(module, grad_in, grad_out):
#     grad_pub.append(grad_out[0].detach())

def show_fmap(features, img, save_name):
    img = mmcv.imread(img)
    h, w, c = img.shape
    for i in range(len(features)):
        fmap = features[i][0]
        fmap = torch.nn.functional.interpolate(fmap, size=[h, w], mode='bilinear')
        
        mean_fmap = torch.max(fmap, dim=1)[0].squeeze()
        mean_fmap /= torch.max(fmap) 
        fmap = mean_fmap * 255
        plt.imshow(img)
        # plt.matshow(mean_fmap.cpu().numpy())
        plt.imshow(fmap.cpu().numpy(), alpha=0.5)
        
        plt.savefig(f'work_dirs/yolox_kaist_3stream_2nc_coattention/{i}_max_' + save_name)
        plt.close()

def decode_offset(model, fmap:dict, pred_res:tuple, img_rgb:str, img_tir:str):
    union_bboxes = pred_res[-2][0]
    rgb_bboxes = pred_res[0][0]
    tir_bboxes = pred_res[1][0]

    debug_dict = model.bbox_head.debug
    valid_mask = debug_dict['pair_bboxes_nms']['valid_mask']
    keep = debug_dict['pair_bboxes_nms']['keep']
    rgb_offset = list(map(lambda x:x.permute(0,2,3,1).reshape(-1, x.shape[1]), fmap['vis']))
    tir_offset = list(map(lambda x:x.permute(0,2,3,1).reshape(-1, x.shape[1]), fmap['inf']))
    rgb_offset = torch.cat(rgb_offset, dim=0)
    tir_offset = torch.cat(tir_offset, dim=0)
    assert len(valid_mask) == len(rgb_offset) == len(tir_offset)
    assert len(keep) == len(union_bboxes)
    
    rgb_offset = rgb_offset[valid_mask][keep]
    tir_offset = tir_offset[valid_mask][keep]

    rgb_x = rgb_offset[:, 1::2]
    rgb_y = rgb_offset[:, ::2]
    tir_x = tir_offset[:, 1::2]
    tir_y = tir_offset[:, ::2]

    w = union_bboxes[:, 2:3] - union_bboxes[:, 0:1]
    h = union_bboxes[:, 3:] - union_bboxes[:, 1:2]

    rgb_x = rgb_x.cpu().numpy() * w + union_bboxes[:, 0:1]
    rgb_y = rgb_y.cpu().numpy() * h + union_bboxes[:, 1:2]
    tir_x = tir_x.cpu().numpy() * w + union_bboxes[:, 0:1]
    tir_y = tir_y.cpu().numpy() * h + union_bboxes[:, 1:2]

    rgb_x_mean = np.mean(rgb_x, axis=1)
    rgb_y_mean = np.mean(rgb_y, axis=1)
    tir_x_mean = np.mean(tir_x, axis=1)
    tir_y_mean = np.mean(tir_y, axis=1)

    img_rgb = mmcv.imread(img_rgb)
    img_tir = mmcv.imread(img_tir)
    width, height, _ = img_rgb.shape

    # for x, y in zip(rgb_x.reshape(-1), rgb_y.reshape(-1)):
    #     cv2.drawMarker(img_rgb, (round(x), round(y)), (0,0,255), markerType=0, markerSize=1)
    # for x, y in zip(tir_x.reshape(-1), tir_y.reshape(-1)):
    #     cv2.drawMarker(img_rgb, (round(x), round(y)), (0,255,0), markerType=0, markerSize=1)
    for bb in union_bboxes.tolist():
        bb = list(map(round, bb))
        cv2.rectangle(img_rgb, tuple(bb[:2]), tuple(bb[2:]), (255,0,0), 1)
    for bb in rgb_bboxes.tolist():
        bb = list(map(round, bb))
        cv2.rectangle(img_rgb, tuple(bb[:2]), tuple(bb[2:]), (0,0,255), 1)
    for bb in tir_bboxes.tolist():
        bb = list(map(round, bb))
        cv2.rectangle(img_rgb, tuple(bb[:2]), tuple(bb[2:]), (0,255,0), 1)
    cv2.imwrite('work_dirs/yolox_kaist_3stream_2nc_coattention/rgb_unpair_m.jpg', img_rgb)


    # for x, y in zip(rgb_x.reshape(-1), rgb_y.reshape(-1)):
    #     cv2.drawMarker(img_tir, (round(x), round(y)), (0,0,255), markerType=0, markerSize=1)
    # for x, y in zip(tir_x.reshape(-1), tir_y.reshape(-1)):
    #     cv2.drawMarker(img_tir, (round(x), round(y)), (0,255,0), markerType=0, markerSize=1)
    for bb in union_bboxes.tolist():
        bb = list(map(round, bb))
        cv2.rectangle(img_tir, tuple(bb[:2]), tuple(bb[2:]), (255,0,0), 1)
    for bb in rgb_bboxes.tolist():
        bb = list(map(round, bb))
        cv2.rectangle(img_tir, tuple(bb[:2]), tuple(bb[2:]), (0,0,255), 1)
    for bb in tir_bboxes.tolist():
        bb = list(map(round, bb))
        cv2.rectangle(img_tir, tuple(bb[:2]), tuple(bb[2:]), (0,255,0), 1)   
    cv2.imwrite('work_dirs/yolox_kaist_3stream_2nc_coattention/tir_unpair_m.jpg', img_tir)

    return


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_vis', help='visible image file')
    parser.add_argument('img_inf', help='infrared image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument('--with_neck', default=False),
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args, fmap_block, input_block, grads):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # model.backbone.stage2.register_forward_hook(forward_vis_hook)
    # model.backbone_lwir.stage2.register_forward_hook(forward_inf_hook)
    # if not args.with_neck:
    #     model.backbone.stage2.register_forward_hook(forward_pub_hook)
    # print(model.neck.out_convs[0])
    
    # model.backbone.stage4_s3.register_forward_hook(forward_pub_hook)
    # model.backbone.stage4_s3.register_backward_hook(backward_pub_hook)
    model.bbox_head.shared_rgb_reg_offset_convs.register_forward_hook(forward_vis_hook)
    model.bbox_head.shared_tir_reg_offset_convs.register_forward_hook(forward_inf_hook)
    # model.bbox_head.rgb_multi_level_reg_convs[0].register_forward_hook(forward_vis_hook)
    # model.bbox_head.rgb_multi_level_reg_convs[1].register_forward_hook(forward_vis_hook)
    # model.bbox_head.rgb_multi_level_reg_convs[2].register_forward_hook(forward_vis_hook)
    # model.bbox_head.tir_multi_level_reg_convs[0].register_forward_hook(forward_inf_hook)
    # model.bbox_head.tir_multi_level_reg_convs[1].register_forward_hook(forward_inf_hook)
    # model.bbox_head.tir_multi_level_reg_convs[2].register_forward_hook(forward_inf_hook)

    model.bbox_head.set_debug('pair_bboxes_nms',[['valid_mask', 'keep']])

    
    if args.img_vis.endswith(('.jpg', '.png')) and args.img_inf.endswith(('.jpg', '.png')):
        # test a single image
        imgs = [args.img_vis, args.img_inf]
        result = inference_rgbt_detector(model, imgs)

        # show_fmap(input_block['vis'], args.img_vis, args.img_vis.split('/')[-1])
        # show_fmap(input_block['inf'], args.img_inf, args.img_inf.split('/')[-1])
        decode_offset(model, fmap_block, result, args.img_vis, args.img_inf)

        def f(x, i):
            return x.permute(0, 3, 1, 2).reshape(1, -1, 64//(2**i), 80//(2**i))
        reg_rgb = [f(fmap_block['pub'][i*4], i) for i in range(3)]
        reg_tir = [f(fmap_block['pub'][i*4+1], i) for i in range(3)]
        # show the results
        # vis backbone
        show_fmap(fmap_block['vis'][0], args.img_vis, args.img_vis.split('/')[-1])
        show_fmap(fmap_block['inf'][0], args.img_inf, args.img_inf.split('/')[-1])
        if not args.with_neck:
            show_fmap(fmap_block['pub'][0], args.img_vis, args.img_vis.split('/')[-1].split('.')[0] + '_pub.jpg')
        show_rgbt_result_pyplot(model, [args.img_vis, args.img_inf], result, score_thr=args.score_thr)
    elif args.img_vis.endswith('.txt') and args.img_inf.endswith('.txt'):
        # test multi images
        img_vis_list = mmcv.list_from_file(args.img_vis)
        img_inf_list = mmcv.list_from_file(args.img_inf)
        assert len(img_vis_list) == len(img_inf_list)
        for i in tqdm(range(len(img_vis_list))):
            imgs = [img_vis_list[i],img_inf_list[i]]
            result = inference_rgbt_detector(model, imgs)
            # show the results
            # vis backbone
            save_dir = 'work_dirs/fmap/'
            show_fmap(fmap_block['vis'][0], img_vis_list[i], save_dir +  img_vis_list[i].split('/')[-1].split('.')[0] + '_vis.jpg')
            show_fmap(fmap_block['inf'][0], img_inf_list[i], save_dir + img_inf_list[i].split('/')[-1].split('.')[0] + '_inf.jpg')
            if not args.with_neck:
                show_fmap(fmap_block['pub'][0], img_vis_list[i], save_dir + img_vis_list[i].split('/')[-1].split('.')[0] + '_pub.jpg')
            # show_rgbt_result_pyplot(model, [img_vis_list[i], img_inf_list[i]], result, score_thr=args.score_thr)
            # delete feature and input tensor
            del fmap_block['vis'][0]
            del fmap_block['inf'][0]
            
            del input_block['vis'][0]
            del input_block['inf'][0]
            if not args.with_neck:
                del fmap_block['pub'][0]
                del input_block['pub'][0]
    else:
        raise ValueError("input dir/img error")
    


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    fmap_vis = list()
    grad_vis = list()
    fmap_inf = list()
    grad_inf = list()
    fmap_pub = list()
    grad_pub = list()
    input_vis = list()
    input_inf = list()
    input_pub = list()
    fmap = {
        'vis':fmap_vis,
        "inf":fmap_inf,
        "pub":fmap_pub
    }
    inputs = {
        'vis':input_vis,
        'inf':input_inf,
        'pub':input_pub
    }
    grads = {
        'vis':grad_vis,
        "inf":grad_inf,
        "pub":grad_pub
    }
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args, fmap, inputs, grads)

