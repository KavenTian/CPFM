config=yolox_kaist_3stream_2nc_coattention
lr=0.01
other=
out=outputs/kaist/MR_results_${config}_${lr}_${other}.txt
for i in {7..18..1}
do
    echo "################# epoch $i #################" >> $out
    bash ./tools/dist_test.sh \
        configs/multispec/${config}.py \
        work_dirs/${config}/epoch_$i.pth \
        4 \
        --format-only \
        --options "jsonfile_prefix=./work_dirs/yolox_csp_kaist/rgbt_det"
    python tools/analysis_tools/json2txt.py
    python ./tools/evaluation_script/evaluation_script.py --rstFiles outputs/kaist/rgbt_det.txt >> $out
done