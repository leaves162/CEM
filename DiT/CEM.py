# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from CEM_utils.DCS_module import DCS_module, gap_flag

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models['DiT-XL/2'](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    state_dict = find_model(args.model_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0
    # Create folder to save samples:
    sample_folder_dir = args.out_path
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()
    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    ours_kwargs = {}
    # duca
    ours_kwargs['cache_type']        = args.cache_type
    ours_kwargs['fresh_ratio']       = args.fresh_ratio
    ours_kwargs['force_fresh']       = args.force_fresh
    ours_kwargs['fresh_threshold']   = args.fresh_threshold
    ours_kwargs['ratio_scheduler']   = args.ratio_scheduler
    ours_kwargs['soft_fresh_weight'] = args.soft_fresh_weight
    # cem
    # ours_kwargs['mode']            = args.mode
    # ours_kwargs['test_FLOPs']      = args.test_FLOPs
    ours_kwargs['prior_path']      = args.prior_path
    ours_kwargs['PRIOR_ERROR_MODELING'] = args.PRIOR_ERROR_MODELING
    ours_kwargs['PEM_C']           = args.PEM_C
    ours_kwargs['DCS_Ns']          = args.DCS_Ns
    ours_kwargs['DCS_interval']    = args.DCS_interval
    ours_kwargs['DCS_weighter']    = args.DCS_weighter
    ours_kwargs['DCS_error_path']  = args.error_path
    ours_kwargs['DCS_timesteps']   = DCS_module(ours_kwargs, args.num_sampling_steps)
    ours_kwargs['DCS_indices']   = gap_flag(ours_kwargs['DCS_timesteps'], args.num_sampling_steps)
    
    for pi in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # kwargs
        model_kwargs['sample_id'] = pi
        if pi==0:
            ours_kwargs['test_FLOPs'] = True
        else:
            ours_kwargs['test_FLOPs'] = False
        model_kwargs.update(ours_kwargs)

        # Sample images:
        if args.ddim_sample:
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        else:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device,
            )
            
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        if not args.PRIOR_ERROR_MODELING:
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--model_path", type=str, default="ckpts/DiT-XL-2/DiT-XL-2-256x256.pt")
    parser.add_argument("--vae_path",  type=str, default="ckpts/DiT-XL-2/vae_ema")
    parser.add_argument("--out_path", type=str, default="DiT/outputs/generations")
    parser.add_argument("--prior_path", type=str, default="DiT/outputs/priors/random_samples", help="if PEM, path for saving sample priors")
    parser.add_argument("--error_path", type=str, default='DiT/outputs/priors/priors.pt', help='if normal generation, path for calling all_caching_C_priors.pt')
    # original parameters
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ddim-sample", action="store_true", default=False)
    # duca
    parser.add_argument("--fresh-ratio", type=float, default=0.07)
    parser.add_argument("--cache-type", type=str, choices=['random', 'attention','similarity','norm', 'compress','kv-norm'], default='random') # only attention supported currently
    parser.add_argument("--ratio-scheduler", type=str, default='ToCa', choices=['linear', 'cosine', 'exp', 'constant','linear-mode','layerwise','ToCa']) #  'ToCa' is the proposed scheduler in Final version of the paper
    parser.add_argument("--force-fresh", type=str, choices=['global', 'local'], default='global', # only global is supported currently, local causes bad results
                        help="Force fresh strategy. global: fresh all tokens. local: fresh tokens acheiving fresh step threshold.")
    parser.add_argument("--fresh-threshold", type=int, default=4) # N in toca
    parser.add_argument("--soft-fresh-weight", type=float, default=0.25, # lambda_3 in toca
                        help="soft weight for updating the stale tokens by adding extra scores.")
    # cem
    # parser.add_argument("--mode", type=str, choices=["duca", "cem"], default="cem")
    # parser.add_argument("--test-FLOPs", action="store_true", default=False, help='Used when calculating flops')
    parser.add_argument("--PRIOR_ERROR_MODELING", action="store_true", default=False, help="Enable prior error modeling")
    parser.add_argument("--PEM_C", type=int, default=1, help="Cache intervals for prior error modeling")
    parser.add_argument("--DCS_Ns", type=int, default=50, help="Number of timesteps for full computation in DSC module")
    parser.add_argument("--DCS_interval", type=int, default=1, help="Interval for dynamic caching strategy")
    parser.add_argument("--DCS_weighter", type=str, choices=["none", "linear", "quadratic"], default="none", help="Weighter for dynamic caching strategy")
    args = parser.parse_args()
    main(args)
