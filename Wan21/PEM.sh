
PRIOR_ROOT="Wan21/outputs/priors/random_samples"
for c in $(seq 2 9); do
  echo "========================================"
  echo "        PEM_C=${c}  (100 samples)       "
  echo "========================================"
  CUDA_VISIBLE_DEVICES=5 python Wan21/CEM.py \
    --PRIOR_ERROR_MODELING \
    --PEM_C "${c}" \
    --prior_path "${PRIOR_ROOT}/C=${c}" \
    --vbench_json_path "Wan21/random_GPT.json" \
    --base_seed 2026
done
echo "Done. Prior tensors: ${PRIOR_ROOT}/C={1..9}/*.pt"
