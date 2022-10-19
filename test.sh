# config=yolox_kaist_3stream_2nc_coattention
# lr=0.01
# other=
# out=outputs/kaist/MR_results_${config}_${lr}_${other}.txt
# for i in {7..18..1}
# do
#     echo "################# epoch $i #################" >> $out
#     bash ./tools/dist_test.sh \
#         configs/multispec/${config}.py \
#         work_dirs/${config}/epoch_$i.pth \
#         4 \
#         --format-only \
#         --options "jsonfile_prefix=./work_dirs/yolox_csp_kaist/rgbt_det"
#     python tools/analysis_tools/json2txt.py
#     python ./tools/evaluation_script/evaluation_script.py --rstFiles outputs/kaist/rgbt_det.txt >> $out
# done

# direction="neg"
# for i in {-10..10..1}
# do 
#     echo "########## Direction=$direction, level=$i ##########"
#     python ./tools/misc/modify_cfg.py --dir $direction --lvl $i
#     bash ./tools/dist_test.sh modified_yolox_kaist_3stream_2nc_coattention.py \
#         work_dirs/yolox_kaist_3stream_2nc_coattention/20221014_003548/epoch_14.pth \
#         2 \
#         --eval bbox \
#         --options jsonfile_prefix=./work_dirs/yolox_kaist_3stream_2nc_coattention/brambox_res
# done
# tar cf "gt_mat_dir_$direction.tar" "annotations/shift_test_matlab/"

bash ./tools/dist_test.sh configs/multispec/yolox_kaist_3stream_2nc_coattention.py \
    work_dirs/yolox_kaist_3stream_2nc_coattention/20221014_003548/epoch_14.pth \
    2 \
    --eval bbox \
    --options jsonfile_prefix=./work_dirs/yolox_kaist_3stream_2nc_coattention/brambox_res
    