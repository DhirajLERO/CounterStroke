##!/bin/bash
#set -e
#
#DATA_ROOT="Imagenet/data/imagenet100"
#LOG_DIR="logs_imagenet100"
#
#mkdir -p "$LOG_DIR"
#
#for start in $(seq 20 20 80); do
#    start2=$((start + 10))
#
#    end1=$((start + 9))
#    end2=$((start2 + 9))
#
#    classes1=$(seq -s, $start $end1)
#    classes2=$(seq -s, $start2 $end2)
#
#    echo "Running:"
#    echo "  GPU0 -> classes ${start}-${end1}"
#    echo "  GPU1 -> classes ${start2}-${end2}"
#
#    CUDA_VISIBLE_DEVICES=0 \
#    python -u analysis/imagenet100/run_blank_graco_pdf.py \
#        --data-root "$DATA_ROOT" \
#        --test-run \
#        --classes "$classes1" \
#        > "${LOG_DIR}/run2_batch_${start}_${end1}.out" 2>&1 &
#
#    pid1=$!
#
#    CUDA_VISIBLE_DEVICES=1 \
#    python -u analysis/imagenet100/run_blank_graco_pdf.py \
#        --data-root "$DATA_ROOT" \
#        --test-run \
#        --classes "$classes2" \
#        > "${LOG_DIR}/run2_batch_${start2}_${end2}.out" 2>&1 &
#
#    pid2=$!
#
#    # Wait for both jobs. If either fails, exit.
#    wait $pid1
#    wait $pid2
#
#    echo "Completed batches ${start}-${end1} and ${start2}-${end2}"
#done
#
#echo "All batches completed successfully."


#!/bin/bash
set -e

DATA_ROOT="Imagenet/data/imagenet100"
LOG_DIR="logs_imagenet100"
GPU=1

mkdir -p "$LOG_DIR"

for start in $(seq 20 10 90); do
    end=$((start + 9))
    classes=$(seq -s, $start $end)

    echo "Running:"
    echo "  GPU${GPU} -> classes ${start}-${end}"

    CUDA_VISIBLE_DEVICES=$GPU \
    python -u analysis/imagenet100/run_blank_graco_pdf.py \
        --data-root "$DATA_ROOT" \
        --test-run \
        --classes "$classes" \
        > "${LOG_DIR}/run2_batch_${start}_${end}.out" 2>&1

    echo "Completed batch ${start}-${end}"
done

echo "All batches completed successfully."