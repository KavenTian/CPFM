import os
from argparse import ArgumentParser

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm

def convert_txt(res_file,
                ann_file,
                out_dir,
                val_list_path,
                 ):
    predict_res = []
    directory = os.path.dirname(out_dir + '/')
    if not os.path.exists(directory):
        print(f'-------------create {out_dir}-------------')
        os.mkdir(directory)

    val_list = []
    with open(val_list_path, 'r') as f:
        for line in f.readlines():

            val_list.append(line.strip('\n'))
    
    cocoGt = COCO(ann_file)
    print("Gt imgs:", len(cocoGt.getImgIds()))
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()

    # write ann with annotaions
    for j, imgid in tqdm(enumerate(imgIds)):
        img_info = cocoDt.loadImgs(imgid)[0]
        res_ann_ids = cocoDt.getAnnIds(imgIds=[imgid])
        res_data = cocoDt.loadAnns(ids=res_ann_ids)
        # 设置txt文件的相关信息
        txt_info = []
        img_prefix = img_info['file_name'][0].split('/')[1].replace('_lwir.png', '')
        txt_path = img_prefix + '.txt'
        txt_ab_path = os.path.join(out_dir, txt_path)
        assert txt_path in val_list, f"{txt_path} not in val list."
        for res in res_data:
            res_bbox = res['bbox']
            for i in range(4):
                if res_bbox[i] < 0:
                    res_bbox[i] = 0
            res_score = res['score']
            txt_info.append(['person', res_bbox[0], res_bbox[1], res_bbox[2] + res_bbox[0], res_bbox[3] + res_bbox[1], res_score])
            predict_res.append([j+1, res_bbox[0], res_bbox[1], res_bbox[2], res_bbox[3], res_score])
        if len(txt_info) > 0:
            txt_info = np.array(txt_info)
            # 保存txt信息
            np.savetxt(txt_ab_path, txt_info, delimiter='',
                        fmt='%s %s %s %s %s %s',
                        )
        else:
            with open(txt_ab_path, 'w') as f: 
                for line_info in txt_info: 
                    f.write(info)
        val_list.remove(txt_path)

    # write txt file
    predict_res = np.array(predict_res)
    np.savetxt("outputs/kaist/rgbt_det.txt", predict_res, delimiter='',fmt='%d,%.4f,%.4f,%.4f,%.4f,%f',)

    # write no annotaions
    for txt in val_list:
        anns = []
        txt_ab_path = os.path.join(out_dir, txt)
        with open(txt_ab_path, 'w') as f:
            for ann in anns:
                f.write(ann)
        
    print('---------------Done!---------------')
def main():
    parser = ArgumentParser(description='json result file convert to txt for MR evaluation')
    parser.add_argument('--result', default='work_dirs/yolox_csp_kaist/rgbt_det.bbox.json',help='result file (json format) path')
    parser.add_argument('--out_dir', default='outputs/kaist/txt_result',help='dir to save analyze result images')
    parser.add_argument('--ann',
                        default='/dataset/KAIST/coco_kaist/annotations/val.json',
                        help='annotation file path'
                        )
    parser.add_argument('--val_list', default='/dataset/KAIST/val_list.txt')
    args = parser.parse_args()
    convert_txt(args.result, args.ann, args.out_dir, args.val_list)

if __name__ == '__main__':
    main()