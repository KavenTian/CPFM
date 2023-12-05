
# file_name="gt_mat_dir_$direction.tar"
# for direction in "x" "y" "pos" "neg"
# do
#     for i in {-10..10..1}
#     do 
#         echo "########## Direction=$direction, level=$i ##########"
#         python ./tools/misc/modify_cfg.py --dir $direction --lvl $i
#         bash ./tools/dist_test.sh modified_yolox_kaist_3stream_2nc_coattention.py \
#             work_dirs/yolox_kaist_3stream_2nc_coattention/20231124_011349/epoch_14.pth \
#             2 \
#             --eval bbox \
#             --options jsonfile_prefix=./work_dirs/yolox_kaist_3stream_2nc_coattention/brambox_res
#         # if [ ! -f $file_name ]
#         # then
#         #     echo "Creat TAR File $file_name"
#         #     tar cf $file_name "annotations/shift_test_matlab/"
#         # else
#         #     tar rf $file_name "annotations/shift_test_matlab/"
#         #     echo "Add New Files to $file_name"
#         # fi
#     done
# done
bash ./tools/dist_test.sh configs/multispec/yolox_kaist_3stream_2nc_coattention.py \
    work_dirs/yolox_kaist_3stream_2nc_coattention/20231124_220548/epoch_14.pth \
    2 \
    --eval bbox \
    --options jsonfile_prefix=./work_dirs/yolox_kaist_3stream_2nc_coattention/brambox_res
    