config=yolox_csp_llvip
epoch=9
bash ./tools/dist_test.sh \
        configs/multispec/${config}.py \
        work_dirs/${config}/epoch_${epoch}.pth \
        4 \
        --out work_dirs/${config}/result_${epoch}.pkl

python tools/analysis_tools/analyze_results.py configs/multispec/${config}.py work_dirs/${config}/result_${epoch}.pkl outputs/kaist/error_analysis/${config}/ --topk 100 --show-score-thr 0.5