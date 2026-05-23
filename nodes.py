"""
ComfyUI-LTXShotRenderer - Simplified nodes for LTX Video Director workflow.

Three nodes replace the 15+ node chain:
  LTXDirector → LTXShotRenderer → LTXShotUpscaler → LTXShotRefiner

Each node is a separate execution step so ComfyUI can manage VRAM:
  - LTXShotRenderer: conditioning + guide + first pass sampling (main model)
  - LTXShotUpscaler: latent upscale only (upscale model, tiny)
  - LTXShotRefiner: re-apply guide + second pass sampling (main model)
"""

import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import node_helpers

try:
    import comfy.nested_tensor
except ImportError:
    comfy.nested_tensor = None


def _get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    if noise_mask is None:
        s = latent["samples"]
        noise_mask = torch.ones(
            (s.shape[0], 1, s.shape[2], s.shape[3], s.shape[4]),
            dtype=s.dtype, device=s.device,
        )
    return noise_mask


def _make_negative(positive):
    """ConditioningZeroOut: zero all tensors for negative conditioning."""
    neg = []
    for p in positive:
        d = p[1].copy()
        if "pooled_output" in d and d["pooled_output"] is not None:
            d["pooled_output"] = torch.zeros_like(d["pooled_output"])
        if "conditioning_lyrics" in d and d["conditioning_lyrics"] is not None:
            d["conditioning_lyrics"] = torch.zeros_like(d["conditioning_lyrics"])
        neg.append((torch.zeros_like(p[0]), d))
    return neg


def _apply_guide(positive, negative, vae, video_latent, guide_data, strength):
    """LTXDirectorGuide: encode guide images and insert into latent."""
    from comfy_extras.nodes_lt import LTXVAddGuide

    latent_image = video_latent["samples"].clone()
    noise_mask = _get_noise_mask(video_latent).clone()
    _, _, latent_length, latent_height, latent_width = latent_image.shape

    if not guide_data:
        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}

    images = guide_data.get("images", [])
    insert_frames = guide_data.get("insert_frames", [])
    strengths = guide_data.get("strengths", [])

    if not images or strength <= 0:
        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}

    scale_factors = vae.downscale_index_formula

    for idx, img_tensor in enumerate(images):
        frame_idx = insert_frames[idx] if idx < len(insert_frames) else 0
        guide_str = strengths[idx] if idx < len(strengths) else strength
        effective_strength = guide_str if guide_str > 0 else strength
        if effective_strength <= 0:
            continue

        _, encoded = LTXVAddGuide.encode(vae, latent_width, latent_height, img_tensor, scale_factors)
        encoded = encoded.to(device=latent_image.device, dtype=latent_image.dtype)

        frame_idx_resolved, _ = LTXVAddGuide.get_latent_index(
            positive, latent_length, img_tensor.shape[0], frame_idx, scale_factors
        )

        positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
            positive, negative, frame_idx_resolved, latent_image, noise_mask,
            encoded, effective_strength, scale_factors
        )
        latent_length = latent_image.shape[2]

    return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


def _sample_pass(model, positive, negative, video_latent, audio_latent,
                 seed, steps, denoise, sampler_name, scheduler_name):
    """Concat AV → sample → separate AV."""
    av_latent = _concat_av(video_latent, audio_latent)
    latent_image = av_latent["samples"]
    noise_mask = av_latent.get("noise_mask", None)
    noise = comfy.sample.prepare_noise(latent_image, seed)

    samples = comfy.sample.sample(
        model, noise, steps, 1.0,
        sampler_name, scheduler_name,
        positive, negative, latent_image,
        denoise=denoise, noise_mask=noise_mask, seed=seed,
    )

    output = {"samples": samples}
    if noise_mask is not None:
        output["noise_mask"] = noise_mask

    return _separate_av(output)


def _concat_av(video_latent, audio_latent):
    """LTXVConcatAVLatent."""
    output = {}
    output.update(video_latent)
    output.update(audio_latent)

    video_mask = video_latent.get("noise_mask", None)
    audio_mask = audio_latent.get("noise_mask", None)
    if video_mask is not None or audio_mask is not None:
        if video_mask is None:
            video_mask = torch.ones_like(video_latent["samples"][:, :1])
        if audio_mask is None:
            audio_mask = torch.ones_like(audio_latent["samples"][:, :1])
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

    output["samples"] = comfy.nested_tensor.NestedTensor((video_latent["samples"], audio_latent["samples"]))
    return output


def _separate_av(av_latent):
    """LTXVSeparateAVLatent."""
    latents = av_latent["samples"].unbind()
    video = av_latent.copy()
    video["samples"] = latents[0]
    audio = av_latent.copy()
    audio["samples"] = latents[1]
    if "noise_mask" in av_latent and av_latent["noise_mask"] is not None:
        masks = av_latent["noise_mask"].unbind()
        video["noise_mask"] = masks[0]
        audio["noise_mask"] = masks[1]
    return video, audio


def _crop_guides(positive, negative, latent):
    """LTXVCropGuides: remove appended keyframes from latent."""
    from comfy_extras.nodes_lt import get_keyframe_idxs

    latent_image = latent["samples"].clone()
    noise_mask = _get_noise_mask(latent)

    _, num_keyframes = get_keyframe_idxs(positive)
    if num_keyframes == 0:
        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}

    latent_image = latent_image[:, :, :-num_keyframes]
    noise_mask = noise_mask[:, :, :-num_keyframes]

    positive = node_helpers.conditioning_set_values(positive, {
        "keyframe_idxs": None, "guide_attention_entries": None,
    })
    negative = node_helpers.conditioning_set_values(negative, {
        "keyframe_idxs": None, "guide_attention_entries": None,
    })
    return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


# =============================================================================
# Node 1: LTXShotRenderer — first pass sampling
# =============================================================================

class LTXShotRenderer:
    """
    First pass: conditioning setup + guide application + sampling + crop.
    Uses main model (~24GB). ComfyUI offloads it after this node completes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
                "guide_data": ("GUIDE_DATA",),
                "vae": ("VAE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 14, "min": 1, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "guide_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("video_latent", "audio_latent", "positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "sampling/ltxv"

    def execute(self, model, positive, video_latent, audio_latent, guide_data, vae,
                frame_rate, seed, steps, sampler_name, scheduler, guide_strength):
        neg_cond = _make_negative(positive)
        pos_cond = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        neg_cond = node_helpers.conditioning_set_values(neg_cond, {"frame_rate": frame_rate})

        pos_g, neg_g, guided_latent = _apply_guide(
            pos_cond, neg_cond, vae, video_latent, guide_data, guide_strength
        )

        video_out, audio_out = _sample_pass(
            model, pos_g, neg_g, guided_latent, audio_latent,
            seed, steps, 1.0, sampler_name, scheduler
        )

        pos_out, neg_out, video_cropped = _crop_guides(pos_g, neg_g, video_out)
        return (video_cropped, audio_out, pos_out, neg_out)


# =============================================================================
# Node 2: LTXShotUpscaler — latent upscale ONLY
# =============================================================================

class LTXShotUpscaler:
    """
    Latent 2x spatial upscale only. Uses the small upscale model (~200MB).
    Separate node so ComfyUI can offload the main model first.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("video_latent",)
    FUNCTION = "execute"
    CATEGORY = "sampling/ltxv"

    def execute(self, video_latent, upscale_model, vae):
        latents = video_latent["samples"]
        original_dtype = latents.dtype
        model_dtype = next(upscale_model.parameters()).dtype
        device = comfy.model_management.get_torch_device()

        comfy.model_management.free_memory(latents.nelement() * 8, device)

        try:
            upscale_model.to(device)
            latents = latents.to(device=device, dtype=model_dtype)
            latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
            upscaled = upscale_model(latents)
            upscaled = vae.first_stage_model.per_channel_statistics.normalize(upscaled)
            upscaled = upscaled.to(device="cpu", dtype=original_dtype)
        finally:
            upscale_model.to("cpu")

        result = video_latent.copy()
        result["samples"] = upscaled
        result.pop("noise_mask", None)
        return (result,)


# =============================================================================
# Node 3: LTXShotRefiner — second pass sampling
# =============================================================================

class LTXShotRefiner:
    """
    Second pass: re-apply guides + partial denoise sampling + crop.
    Uses main model again. ComfyUI reloads it after upscaler is offloaded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
                "guide_data": ("GUIDE_DATA",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "denoise": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "execute"
    CATEGORY = "sampling/ltxv"

    def execute(self, model, positive, negative, video_latent, audio_latent,
                guide_data, vae, seed, steps, denoise, sampler_name, scheduler,
                guide_strength):
        pos_up, neg_up, guided_up = _apply_guide(
            positive, negative, vae, video_latent, guide_data, guide_strength
        )

        video_out, audio_out = _sample_pass(
            model, pos_up, neg_up, guided_up, audio_latent,
            seed, steps, denoise, sampler_name, scheduler
        )

        _, _, video_final = _crop_guides(pos_up, neg_up, video_out)
        return (video_final, audio_out)


NODE_CLASS_MAPPINGS = {
    "LTXShotRenderer": LTXShotRenderer,
    "LTXShotUpscaler": LTXShotUpscaler,
    "LTXShotRefiner": LTXShotRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXShotRenderer": "LTX Shot Renderer",
    "LTXShotUpscaler": "LTX Shot Upscaler",
    "LTXShotRefiner": "LTX Shot Refiner",
}
