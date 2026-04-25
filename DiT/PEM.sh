#!/usr/bin/env bash

### CEM: Prior Error Modeling (PEM) for prior sampling (100 samples)

PRIOR_ROOT="/nas/st/Apps/code/CEM/DiT/outputs/priors/random_samples"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
for c in $(seq 1 9); do
  echo "========================================"
  echo "        PEM_C=${c}  (100 samples)       "
  echo "========================================"
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE:-1}" \
    --master_port=$((MASTER_PORT_BASE + c)) \
    /nas/st/Apps/code/CEM/DiT/CEM.py \
    --mode cem \
    --PRIOR_ERROR_MODELING \
    --PEM_C "${c}" \
    --prior_path "${PRIOR_ROOT}/C=${c}" \
    --num-fid-samples 100 \
    --per-proc-batch-size 1 \
    --num-sampling-steps 50 \
    --global-seed 2026
done
echo "Done. Prior tensors: ${PRIOR_ROOT}/C={1..9}/*.pt"