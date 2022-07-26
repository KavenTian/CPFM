config=yolox3lca_distill_yoloxl_kaist
mkdir work_dirs/${config}_stu
for i in {1..15..1}
do
    echo "################# epoch $i #################"
    python tools/pth_transfer.py --fgd_path work_dirs/${config}/epoch_${i}.pth --output_path work_dirs/${config}_stu/epoch_${i}.pth
done