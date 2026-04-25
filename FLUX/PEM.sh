
### CEM: Prior Error Modeling (PEM) for prior sampling (100 samples)

PRIOR_ROOT="FLUX/outputs/priors/random_samples"
for c in $(seq 2 9); do
  echo "========================================"
  echo "        PEM_C=${c}  (100 samples)       "
  echo "========================================"
  CUDA_VISIBLE_DEVICES=4 python FLUX/CEM.py \
    --PRIOR_ERROR_MODELING \
    --PEM_C "${c}" \
    --prior_path "${PRIOR_ROOT}/C=${c}" \
    --prompt_file "FLUX/random_GPT.txt" \
    --seed 2026
done
echo "Done. Prior tensors: ${PRIOR_ROOT}/C={1..9}/*.pt"