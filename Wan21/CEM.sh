
ERROR_ROOT="Wan21/outputs/priors/priors.pt"
BASE_OUT_ROOT="Wan21/outputs/generations"

DCS_Ns=9 # full compt steps, equal to TaylorSeer n=6
DCS_interval=1
DCS_weighter=quadratic

OUT_ROOT="${BASE_OUT_ROOT}/CEM_Ns_${DCS_Ns}_interval_${DCS_interval}_${DCS_weighter}"

echo -e "\033[31m CEM with Ns=${DCS_Ns}, interval=${DCS_interval}, weighter=${DCS_weighter} \033[0m"
echo -e "\033[31m Generation start: out_dir: ${OUT_ROOT} \033[0m"

CUDA_VISIBLE_DEVICES=5 python Wan21/CEM.py \
    --error_path "${ERROR_ROOT}" \
    --out_path "${OUT_ROOT}" \
    --DCS_Ns "${DCS_Ns}" \
    --DCS_interval "${DCS_interval}" \
    --DCS_weighter "${DCS_weighter}"

echo -e "\033[31m CEM with Ns=${DCS_Ns}, interval=${DCS_interval}, weighter=${DCS_weighter} \033[0m"
echo -e "\033[31m Generation end: out_dir: ${OUT_ROOT} \033[0m"
