#!/usr/bin/env bash
# run_all.sh  ── 按 epoch 1-28 循环：
#   ① 把 235-243 行注释掉 → torchrun save_tfi_and_gray.py
#   ② 恢复原文件 → bash valid.sh（取最后 6 行写日志）
# 日志：all_metrics_YYYYMMDD_HHMMSS.log

set -euo pipefail

SEQ_PY="dataset/x4k1000fps_sequence.py"
BACKUP="${SEQ_PY}.orig"                  # 原始备份，只做一次
LOG="all_metrics_$(date '+%Y%m%d_%H%M%S').log"

# 0️⃣ 备份一次原文件
if [[ ! -f "$BACKUP" ]]; then
    cp "$SEQ_PY" "$BACKUP"
fi

echo "=============== 批量验证开始 $(date) ===============" | tee "$LOG"

for n in $(seq 29 39); do
    epoch=$(printf "%03d" "$n")
    CKPT="exp_fusion/ckpt_epoch${epoch}.pth"
    echo -e "\n>>> [Epoch ${epoch}] 开始 —— $(date)" | tee -a "$LOG"

    if [[ ! -f "$CKPT" ]]; then
        echo "[警告] 找不到 $CKPT，跳过。" | tee -a "$LOG"
        continue
    fi

    # 1️⃣ 生成“注释版”并替换
    cp "$BACKUP" "$SEQ_PY"                            # 先恢复为原版
    sed -i '235,243s/^/#/' "$SEQ_PY"                  # 注释 235-243 行

    # 2️⃣ torchrun（完整输出进日志）
    torchrun --nproc_per_node=2 save_tfi_and_gray.py \
        --dataset_root /gpfsdata/home/lvjiahui/projects/x4k1000fps/ \
        --checkpoint   "$CKPT" \
        --output_dir   /gpfsdata/home/lvjiahui/projects/x4k1000fps/ \
        --length_spike 33  --crop 256 \
        --batch_size   200 --n_workers 8 \
        --use_blur \
        2>&1 | tee -a "$LOG"

    # 3️⃣ 恢复“未注释版”
    cp "$BACKUP" "$SEQ_PY"

    # 4️⃣ valid.sh（仅记录最后 6 行）
    echo "-- 运行 valid.sh --" | tee -a "$LOG"
    tmp_valid=$(mktemp)
    bash valid.sh 2>&1 | tee "$tmp_valid"
    tail -n 6 "$tmp_valid" | tee -a "$LOG"
    rm -f "$tmp_valid"

    echo "<<< [Epoch ${epoch}] 结束 —— $(date)" | tee -a "$LOG"
done

echo -e "\n=============== 全部完成 $(date) ===============" | tee -a "$LOG"
