distill_config=yolox3lca_distill_yoloxl_kaist
stu_config=yolox_csp_pub_kaist
other=stu
out=outputs/kaist/MR_results_${stu_config}_${other}.txt
for i in {6..15..1}
do
    echo "################# epoch $i #################" >> $out
    bash ./tools/dist_test.sh \
        configs/multispec/${stu_config}.py \
        work_dirs/${distill_config}_stu/epoch_$i.pth \
        4 \
        --format-only \
        --options "jsonfile_prefix=./work_dirs/${distill_config}_stu/rgbt_det"
    python tools/analysis_tools/json2txt.py --result work_dirs/${distill_config}_stu/rgbt_det.bbox.json
    python ./tools/evaluation_script/evaluation_script.py --rstFiles outputs/kaist/rgbt_det.txt >> $out
done