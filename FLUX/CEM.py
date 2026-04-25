from typing import Any, Dict, Optional, Tuple, Union
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from collections import deque
from CEM_utils.DCS_module import DCS_module, DCS_module_interval_gaps


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        if self.enable_teacache:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
            # if self.cnt == 0 or self.cnt == self.num_steps-1:
            #     should_calc = True
            #     self.accumulated_rel_l1_distance = 0
            # else: 
            #     coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            #     rescale_func = np.poly1d(coefficients)
            #     self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            #     if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
            #         should_calc = False
            #     else:
            #         should_calc = True
            #         self.accumulated_rel_l1_distance = 0
            
            # cem: dcs module to replace the teacache
            if self.cnt in self.DCS_timesteps:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                should_calc = False

            self.previous_modulated_input = modulated_inp
        prior_step_idx = self.cnt
        # print(self.cnt, should_calc)
        self.cnt += 1 
        if self.cnt == self.num_steps:
            self.cnt = 0
        
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for index_block, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                for index_block, block in enumerate(self.single_transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)
                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        hidden_states = block(
                            hidden_states=hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )

                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                self.previous_residual = hidden_states - ori_hidden_states
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                if self.PRIOR_ERROR_MODELING and index_block == len(self.transformer_blocks) - 1:
                    with torch.no_grad():
                        # torch.Size([1, 512, 3072]) torch.Size([1, 4096, 3072])
                        # print('double:', index_block, encoder_hidden_states.shape, hidden_states.shape)
                        tmp_feat = torch.cat((encoder_hidden_states, hidden_states), dim=1).mean(dim=1).detach()
                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                
                if self.PRIOR_ERROR_MODELING and index_block == len(self.single_transformer_blocks) - 1:
                    with torch.no_grad():
                        # torch.Size([1, 4608, 3072])
                        # print('single:', index_block, hidden_states.shape)
                        tmp_feat2 = hidden_states.mean(dim=1).detach()
                        curre_feat = (tmp_feat + tmp_feat2) / 2
                        curre_stored = curre_feat.detach().clone()
                        if prior_step_idx < self.cache_dic['PEM_C']:
                            self.cache_dic['prior_cache'].append(curre_stored)
                        else:
                            cache_feat = self.cache_dic['prior_cache'].popleft()
                            a = cache_feat.reshape(-1).float()
                            b = curre_feat.reshape(-1).float()
                            denom = a.norm() * b.norm() + 1e-8
                            cos_sim = (torch.dot(a, b) / denom).clamp(-1.0, 1.0)
                            self.cache_dic['prior_errors'][prior_step_idx] = 1.0 - cos_sim
                            self.cache_dic['prior_cache'].append(curre_stored)
                        if prior_step_idx == self.num_steps - 1:
                            # print(self.cache_dic['prior_errors'])
                            os.makedirs(self.cache_dic['prior_path'], exist_ok=True)
                            torch.save(self.cache_dic['prior_errors'].detach().cpu(), f"{self.cache_dic['prior_path']}/{self.sample_id}.pt")

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Teacache for flux")
    parser.add_argument('--prompt_file', type=str, default='DrawBench200.txt')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance value.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images per prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (prompt batching).')
    parser.add_argument('--add_sampling_metadata', action='store_true', help='Whether to add prompt metadata to images.')
    parser.add_argument('--use_nsfw_filter', action='store_true', help='Enable NSFW filter.')
    # teacache
    parser.add_argument('--rel_l1_thresh', type=float, default=0.6)
    # cem
    parser.add_argument("--out_path", type=str, default="FLUX/outputs/generations/debug")
    parser.add_argument("--prior_path", type=str, default="", help="if PEM, path for saving sample priors")
    parser.add_argument("--error_path", type=str, default="", help='if normal generation, path for calling all_caching_C_priors.pt')
    parser.add_argument("--PRIOR_ERROR_MODELING", action="store_true", default=False, help="Enable prior error modeling")
    parser.add_argument("--PEM_C", type=int, default=1, help="Cache intervals for prior error modeling")
    parser.add_argument("--DCS_Ns", type=int, default=50, help="Number of timesteps for full computation in DSC module")
    parser.add_argument("--DCS_interval", type=int, default=1, help="Interval for dynamic caching strategy")
    parser.add_argument("--DCS_weighter", type=str, choices=["none", "linear", "quadratic"], default="none",  help="Weighter for dynamic caching strategy")
    args = parser.parse_args()

    FluxTransformer2DModel.forward = teacache_forward
    pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    # overload
    
    enable_teacache = True
    if args.PRIOR_ERROR_MODELING:
        enable_teacache = False
    pipeline.transformer.__class__.enable_teacache = enable_teacache
    pipeline.transformer.__class__.cnt = 0
    pipeline.transformer.__class__.num_steps = args.num_steps
    pipeline.transformer.__class__.rel_l1_thresh = args.rel_l1_thresh # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    pipeline.transformer.__class__.previous_modulated_input = None
    pipeline.transformer.__class__.previous_residual = None
    # cem
    pipeline.transformer.__class__.PRIOR_ERROR_MODELING = args.PRIOR_ERROR_MODELING
    # cem: prior error modeling
    if args.PRIOR_ERROR_MODELING:
        cache_dic = {}
        cache_dic['prior_path']           = args.prior_path
        cache_dic['prior_errors']         = torch.zeros(args.num_steps)
        cache_dic['prior_cache']          = deque()
        cache_dic['PEM_C']                = args.PEM_C
        pipeline.transformer.__class__.cache_dic = cache_dic
    # cem: dynamic caching strategy
    cem_dict = {
        'DCS_Ns': args.DCS_Ns,
        'DCS_interval': args.DCS_interval,
        'DCS_weighter': args.DCS_weighter,
        'DCS_error_path': args.error_path,
    }
    # pipeline.transformer.__class__.DCS_timesteps = DCS_module(cem_dict, args.num_steps)
    # for example: warm up + Error correction stage with lower intervals
    pipeline.transformer.__class__.DCS_timesteps = DCS_module_interval_gaps(cem_dict, args.num_steps, [(0,2,(1,2,)),(10,18,(2,)),(20,36,(2,))])

    pipeline.to("cuda")
    prompts = read_prompts(args.prompt_file)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        pipeline.transformer.__class__.sample_id = idx
        img = pipeline(
            prompt,
            num_inference_steps=args.num_steps,
            generator=torch.Generator("cpu").manual_seed(args.seed)
            ).images[0]
        img.save(f"{args.out_path}/img_{idx}.png")

if __name__ == '__main__':
    main()