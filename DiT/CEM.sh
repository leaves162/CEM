#!/usr/bin/env bash

### CEM: Plug-and-Play Deployment (PPD) for DiT generation acceleration

ERROR_ROOT="DiT/outputs/priors/priors.pt"
BASE_OUT_ROOT="DiT/outputs/generations"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29501}"

DCS_Ns=10 # full compt steps, equal to DuCa N=6
DCS_interval=1 # candidate cache intervals
DCS_weighter=quadratic # add weights in prior errors
OUT_ROOT="${BASE_OUT_ROOT}/CEM_Ns_${DCS_Ns}_interval_${DCS_interval}_${DCS_weighter}"

echo -e "\033[31m CEM with Ns=${DCS_Ns}, interval=${DCS_interval}, weighter=${DCS_weighter} \033[0m"
echo -e "\033[31m Generation start: out_dir: ${OUT_ROOT} \033[0m"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    --master_port="${MASTER_PORT_BASE}" \
    /nas/st/Apps/code/CEM/DiT/CEM.py \
    --error_path "${ERROR_ROOT}" \
    --out_path "${OUT_ROOT}" \
    --num-fid-samples 50000 \
    --per-proc-batch-size 32 \
    --num-sampling-steps 50 \
    --image-size 256 \
    --num-classes 1000 \
    --cfg-scale 1.5 \
    --global-seed 0 \
    --ddim-sample \
    --cache-type attention \
    --fresh-ratio 0.05 \
    --ratio-scheduler ToCa \
    --force-fresh global \
    --fresh-threshold 1 \
    --soft-fresh-weight 0.25 \
    --DCS_Ns "${DCS_Ns}" \
    --DCS_interval "${DCS_interval}" \
    --DCS_weighter "${DCS_weighter}"

echo -e "\033[31m CEM with Ns=${DCS_Ns}, interval=${DCS_interval}, weighter=${DCS_weighter} \033[0m"
echo -e "\033[31m Generation end: out_dir: ${OUT_ROOT} \033[0m"