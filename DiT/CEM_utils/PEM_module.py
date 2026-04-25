import os
import torch
import argparse

def main(args):
    prior_path = args.prior_path
    prior_sample_path = f"{prior_path}/random_samples"
    C_idxs = [1,2,3,4,5,6,7,8,9]
    all_C_priors = {}
    for C_idx in C_idxs:
        C_file = f"{prior_sample_path}/C={C_idx}"
        # c=i file list
        C_file_list = os.listdir(C_file)
        curr_C_priors = []
        for f in C_file_list:
            prior = torch.load(f"{C_file}/{f}")
            curr_C_priors.append(prior)
        # c=i priors stack
        curr_C_priors = torch.stack(curr_C_priors)
        curr_C_priors = torch.mean(curr_C_priors, dim=0)
        print('C=',C_idx, 'prior:', curr_C_priors)
        all_C_priors[f"C={C_idx}"] = curr_C_priors
    torch.save(all_C_priors, f"{prior_path}/priors.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--prior_path", type=str, default="DiT/outputs/priors")
    args = parser.parse_args()
    main(args)