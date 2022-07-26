# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference, inference_rgbt_detector,
                        init_detector, show_result_pyplot, show_rgbt_result_pyplot)
import mmcv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def forward_vis_hook(module, data_input, data_output):
    fmap_vis.append(data_output)
    input_vis.append(data_input)

def forward_inf_hook(module, data_input, data_output):
    fmap_inf.append(data_output)
    input_inf.append(data_input)

def forward_pub_hook(module, data_input, data_output):
    fmap_pub.append(data_output)
    input_pub.append(data_input)

def show_fmap(fmap, img, save_name):
    img = mmcv.imread(img)
    h, w, c = img.shape
    fmap = torch.nn.functional.interpolate(fmap, size=[h, w], mode='bilinear')
    
    mean_fmap = torch.mean(fmap, dim=1).squeeze()
    mean_fmap /= torch.max(fmap) 
    fmap = mean_fmap * 255
    plt.imshow(img)
    # plt.matshow(mean_fmap.cpu().numpy())
    plt.imshow(fmap.cpu().numpy(), alpha=0.5)
    
    plt.savefig(save_name)
    plt.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_vis', help='visible image file')
    parser.add_argument('img_inf', help='infrared image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--with_neck', default=False),
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args, fmap_block, input_block):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.backbone.stage2.register_forward_hook(forward_vis_hook)
    model.backbone_lwir.stage2.register_forward_hook(forward_inf_hook)
    if not args.with_neck:
        model.backbone.stage2.register_forward_hook(forward_pub_hook)

    # print(model.neck.out_convs[0])

    
    if args.img_vis.endswith(('.jpg', '.png')) and args.img_inf.endswith(('.jpg', '.png')):
        # test a single image
        imgs = [args.img_vis, args.img_inf]
        result = inference_rgbt_detector(model, imgs)
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
            save_dir = 'outputs/llvip/fmap/'
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
    fmap_inf = list()
    fmap_pub = list()
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
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args, fmap, inputs)

