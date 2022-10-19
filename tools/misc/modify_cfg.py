import argparse
from locale import nl_langinfo
import os

from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description='Modify An Existed Config File')
    parser.add_argument('--config', default='configs/multispec/yolox_kaist_3stream_2nc_coattention.py')
    parser.add_argument('--dir', type=str, default='x', help="x, y, neg, pos")
    parser.add_argument('--lvl', type=int, default=0)

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    new_cfg = 'modified_' + args.config.split('/')[-1]
    assert args.dir in ['x', 'y', 'neg', 'pos']

    if args.dir == 'x':
        _list = [args.lvl, 0]
    elif args.dir == 'y':
        _list = [0, args.lvl]
    elif args.dir == 'pos':
        _list = [args.lvl, args.lvl]
    else:
        _list = [args.lvl, -1 * args.lvl]
    
       
    with open(args.config, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:  # 217
        if 'new_shifts=[' in line.replace(' ', ''):
            new_l = line.split('=')[0] + f' = {str(_list)}\n'
            new_lines.append(new_l)
        elif '..' in line:
            new_l = line.replace('..', 'configs')
            new_lines.append(new_l)
        else:
            new_lines.append(line)
    out = ''.join(new_lines)

    with open(new_cfg, 'w') as f:
        f.write(out)

    print(f"Got direction: {args.dir}... Level: {args.lvl}\n")


if __name__ == '__main__':
    main()