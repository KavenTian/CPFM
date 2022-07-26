config=yolox_csp_kaist_2nc
epoch=10
bash ./tools/dist_test.sh \
        configs/multispec/${config}.py \
        work_dirs/${config}/epoch_${epoch}.pth \
        4 \
        --show-dir outputs/kaist/img_results/${config}/