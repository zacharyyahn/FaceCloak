import torch
import numpy as np
import gc

def preprocess_tanh(current_im):
    tanh_constant = 2 - 1e-6
    current_im /= 255.0
    current_im -= 0.5
    current_im *= tanh_constant
    current_im = torch.tanh(current_im)
    return current_im

def reverse_tanh(current_im):
    tanh_constant = 2 - 1e-6
    current_im = np.clip(current_im, a_min=-1+1e-6, a_max=1-1e-6) # make sure we don't cause any infinite values
    current_im = (np.arctanh(current_im) / tanh_constant + 0.5) * 255.0
    #print("Null or inf values in reverse_tanh:", np.all(np.isinf(im)), np.all(np.isnan(im)))
    return current_im

def preprocess_divide(current_im):
    current_im = (current_im - 127.5) / 128.0
    return current_im

def reverse_divide(h_current_im):
    h_current_im = np.clip(h_current_im, -1, 1)
    h_current_im = (h_current_im * 128.0) + 127.5
    h_current_im = np.clip(h_current_im, 0, 255.0)
    return h_current_im

def do_nothing(im):
    return im

@torch.no_grad()
def _make_negative_embeds(pipe, batch_size, like_embed):
    """
    Create negative (unconditional) prompt embeds that match shape/dtype/device
    of Arc2Face prompt embeds. Uses the Arc2Face text encoder on "".
    Falls back to zeros if tokenizer/encoder aren't available.
    """
    try:
        tok = pipe.tokenizer(
            [""] * batch_size,
            return_tensors="pt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length
        ).to(like_embed.device)
        # Arc2Face CLIPTextModelWrapper returns (hidden_states, ...) so take [0]
        neg = pipe.text_encoder(tok.input_ids)[0]
        # Make sure dtype matches UNet (usually fp16)
        neg = neg.to(dtype=like_embed.dtype)
    except Exception:
        neg = torch.zeros_like(like_embed)
    return neg

def pipeline_forward_with_grad(
    pipe,
    prompt_embeds,                 # [B, T, C] from project_face_embs(...), requires_grad=True
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=None,
    latents=None,
    negative_prompt_embeds=None,   # optional [B, T, C]; if None we build one
):
    device = pipe._execution_device
    dtype  = pipe.unet.dtype
    bsz    = prompt_embeds.shape[0]

    # Build negative embeds if not provided (match shape/dtype/device)
    if negative_prompt_embeds is None:
        negative_prompt_embeds = _make_negative_embeds(pipe, bsz, prompt_embeds).detach().contiguous()

    # Scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Latent init
    if latents is None:
        latents = torch.randn(
            (bsz, pipe.unet.in_channels, height // 8, width // 8),
            generator=generator, device=device, dtype=dtype
        )
    # Match HF pipeline: scale init noise
    latents = latents * pipe.scheduler.init_noise_sigma

    # Denoising loop (single UNet call per step with concatenation for CFG)
    for t in timesteps:
        # Some schedulers require scaling the input latents each step
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)

        # Concatenate unconditional + conditional for a single forward
        latent_in   = torch.cat([latent_model_input, latent_model_input], dim=0)
        embeds_in   = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # One UNet call
        noise_pred  = pipe.unet(latent_in, t, encoder_hidden_states=embeds_in).sample

        # Split and apply CFG
        noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Right after the denoising loop ends

    # Decode (match HF: divide BEFORE decode)
    image_pt = pipe.vae.decode(
        latents / getattr(pipe.vae.config, "scaling_factor", 0.18215),
        return_dict=False
    )[0]

    # Rescale to [0,1]
    image_pt = (image_pt / 2 + 0.5).clamp(0, 1)

    # Detach only for visualization/output
    image_pil = pipe.image_processor.postprocess(image_pt.detach(), output_type="pt")[0]

    # Cleanup to prevent Jupyter re-run OOMs
    del noise_pred, noise_uncond, noise_text, latent_in, embeds_in, latent_model_input
    gc.collect()
    torch.cuda.empty_cache()

    return image_pt, 2 * (image_pil.cpu().float().detach().numpy().transpose((1, 2, 0)) - 0.5)
