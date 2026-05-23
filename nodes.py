"""
ComfyUI-LTXShotRenderer - Simplified nodes for LTX Video Director workflow.

Reduces the 15+ node chain down to 2 nodes:
  LTXDirector → LTXShotRenderer

LTXShotRenderer internally handles:
  - ConditioningZeroOut + LTXVConditioning (frame rate)
  - LTXDirectorGuide (apply guide images to conditioning/latent)
  - LTXVConcatAVLatent (merge video + audio)
  - CFGGuider + BasicScheduler + SamplerCustomAdvanced (sampling)
  - LTXVSeparateAVLatent (split output)
  - LTXVCropGuides (remove guide frames)
  - Optional: LTXVLatentUpsampler + second pass
"""

import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import node_helpers

try:
    import comfy.nested_tensor
except ImportError:
    comfy.nested_tensor = None


def get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    if noise_mask is None:
        noise_mask = torch.ones(
            (latent["samples"].shape[0], 1, latent["samples"].shape[2],
             latent["samples"].shape[3], latent["samples"].shape[4]),
            dtype=latent["samples"].dtype, device=latent["samples"].device
        )
    return noise_mask


class LTXShotRenderer:
    """
    All-in-one sampler for LTX Director workflow.

    Input: model, positive conditioning, video_latent, audio_latent, guide_data from LTXDirector.
    Output: Final video_latent and audio_latent ready for VAE decode.

    Internally performs two-pass sampling:
      Pass 1: Apply guides → concat AV → sample → separate → crop guides
      Pass 2 (optional): Crop → upscale → re-apply guides → concat AV → sample → separate → crop
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
                "guide_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                                             "tooltip": "Strength for guide images in first pass"}),
            },
            "optional": {
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "upscale_steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "upscale_denoise": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                                     "tooltip": "Strength for guide images in upscale pass"}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent", "first_pass_video")
    FUNCTION = "execute"
    CATEGORY = "sampling/ltxv"

    def execute(
        self, model, positive, video_latent, audio_latent, guide_data, vae,
        frame_rate, seed, steps, sampler_name, scheduler, guide_strength,
        upscale_model=None, upscale_steps=4, upscale_denoise=0.42, upscale_guide_strength=1.0,
    ):
        # === Step 1: Build positive/negative conditioning with frame_rate ===
        neg_cond = self._make_negative(positive)
        pos_cond = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        neg_cond = node_helpers.conditioning_set_values(neg_cond, {"frame_rate": frame_rate})

        # === Step 2: Apply Director Guide (first pass) ===
        pos_g, neg_g, guided_latent = self._apply_guide(
            pos_cond, neg_cond, vae, video_latent, guide_data, guide_strength
        )

        # === Step 3: First pass sampling (cfg=1, denoise=1.0 — full generation) ===
        video_out, audio_out = self._sample_pass(
            model, pos_g, neg_g, guided_latent, audio_latent,
            seed, steps, 1.0, sampler_name, scheduler
        )

        # === Step 4: Crop guides from first pass result ===
        pos_cropped, neg_cropped, video_cropped = self._crop_guides(pos_g, neg_g, video_out)
        first_pass_result = {"samples": video_cropped["samples"].clone()}

        # === Step 5: Optional upscale + second pass ===
        if upscale_model is not None and upscale_steps > 0 and upscale_denoise > 0:
            video_upscaled = self._latent_upscale(video_cropped, upscale_model, vae)

            # Re-apply guide at upscaled resolution
            pos_up, neg_up, guided_up = self._apply_guide(
                pos_cropped, neg_cropped, vae, video_upscaled, guide_data, upscale_guide_strength
            )

            # Second pass sampling (cfg=1, partial denoise for refinement)
            video_final, audio_final = self._sample_pass(
                model, pos_up, neg_up, guided_up, audio_out,
                seed, upscale_steps, upscale_denoise, sampler_name, scheduler
            )

            # Crop guides from final output
            _, _, video_final = self._crop_guides(pos_up, neg_up, video_final)
        else:
            video_final = video_cropped
            audio_final = audio_out

        return (video_final, audio_final, first_pass_result)

    def _make_negative(self, positive):
        """Create zeroed-out negative conditioning from positive (ConditioningZeroOut)."""
        neg = []
        for p in positive:
            d = p[1].copy()
            if "pooled_output" in d and d["pooled_output"] is not None:
                d["pooled_output"] = torch.zeros_like(d["pooled_output"])
            if "conditioning_lyrics" in d and d["conditioning_lyrics"] is not None:
                d["conditioning_lyrics"] = torch.zeros_like(d["conditioning_lyrics"])
            neg.append((torch.zeros_like(p[0]), d))
        return neg

    def _apply_guide(self, positive, negative, vae, video_latent, guide_data, strength):
        """Apply LTXDirectorGuide: encode guide images into latent at their frame positions.

        guide_data is a dict with keys: "images" (list of tensors), "insert_frames" (list of ints),
        "strengths" (list of floats).
        """
        from comfy_extras.nodes_lt import LTXVAddGuide

        latent_image = video_latent["samples"].clone()
        noise_mask = get_noise_mask(video_latent).clone()
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

            # Use the overall strength as a multiplier
            effective_strength = guide_str if guide_str > 0 else strength
            if effective_strength <= 0:
                continue

            # Encode image through VAE
            _, encoded = LTXVAddGuide.encode(vae, latent_width, latent_height, img_tensor, scale_factors)

            # Ensure encoded is on the same device as latent_image
            encoded = encoded.to(device=latent_image.device, dtype=latent_image.dtype)

            # Calculate latent frame index
            frame_idx_resolved, latent_idx = LTXVAddGuide.get_latent_index(
                positive, latent_length, img_tensor.shape[0], frame_idx, scale_factors
            )

            # Append keyframe to latent and conditioning
            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx_resolved, latent_image, noise_mask,
                encoded, effective_strength, scale_factors
            )

            latent_length = latent_image.shape[2]

        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}

    def _sample_pass(self, model, positive, negative, video_latent, audio_latent,
                     seed, steps, denoise, sampler_name, scheduler_name):
        """Run one sampling pass: concat AV → sample → separate AV.

        Uses comfy.sample.sample which is the same path as the built-in KSampler node.
        """
        # Concat video + audio
        av_latent = self._concat_av(video_latent, audio_latent)

        latent_image = av_latent["samples"]
        noise_mask = av_latent.get("noise_mask", None)

        # Generate noise matching the latent structure
        noise = comfy.sample.prepare_noise(latent_image, seed)

        # Use the high-level sample() which handles KSampler creation,
        # sigmas, device management, and model patching internally
        samples = comfy.sample.sample(
            model, noise, steps, 1.0,  # cfg=1.0 always for LTX Director
            sampler_name, scheduler_name,
            positive, negative,
            latent_image,
            denoise=denoise,
            noise_mask=noise_mask,
            seed=seed,
        )

        output = {"samples": samples}
        if noise_mask is not None:
            output["noise_mask"] = noise_mask

        # Separate AV
        return self._separate_av(output)

    def _concat_av(self, video_latent, audio_latent):
        """Merge video and audio latents (LTXVConcatAVLatent)."""
        output = {}
        output.update(video_latent)
        output.update(audio_latent)

        video_noise_mask = video_latent.get("noise_mask", None)
        audio_noise_mask = audio_latent.get("noise_mask", None)

        if video_noise_mask is not None or audio_noise_mask is not None:
            if video_noise_mask is None:
                video_noise_mask = torch.ones_like(video_latent["samples"][:, :1])
            if audio_noise_mask is None:
                audio_noise_mask = torch.ones_like(audio_latent["samples"][:, :1])
            output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))

        output["samples"] = comfy.nested_tensor.NestedTensor((video_latent["samples"], audio_latent["samples"]))
        return output

    def _separate_av(self, av_latent):
        """Split AV latent into video + audio (LTXVSeparateAVLatent)."""
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

    def _crop_guides(self, positive, negative, latent):
        """Remove appended guide keyframes from latent (LTXVCropGuides)."""
        from comfy_extras.nodes_lt import get_keyframe_idxs

        latent_image = latent["samples"].clone()
        noise_mask = get_noise_mask(latent)

        _, num_keyframes = get_keyframe_idxs(positive)
        if num_keyframes == 0:
            return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}

        latent_image = latent_image[:, :, :-num_keyframes]
        noise_mask = noise_mask[:, :, :-num_keyframes]

        positive = node_helpers.conditioning_set_values(positive, {
            "keyframe_idxs": None,
            "guide_attention_entries": None,
        })
        negative = node_helpers.conditioning_set_values(negative, {
            "keyframe_idxs": None,
            "guide_attention_entries": None,
        })

        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}

    def _latent_upscale(self, video_latent, upscale_model, vae):
        """Upscale video latent using latent upscale model (LTXVLatentUpsampler).

        Matches ComfyUI's LTXVLatentUpsampler: handles dtype conversion,
        device management, and VAE per-channel statistics normalization.
        """
        latents = video_latent["samples"]
        original_dtype = latents.dtype

        # Determine model dtype and device
        model_dtype = next(upscale_model.parameters()).dtype
        device = comfy.model_management.get_torch_device()

        # Free memory before upscale
        comfy.model_management.free_memory(latents.nelement() * 8, device)

        try:
            upscale_model.to(device)

            # Convert to model dtype, un-normalize, upscale, re-normalize
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

        return result


NODE_CLASS_MAPPINGS = {
    "LTXShotRenderer": LTXShotRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXShotRenderer": "LTX Shot Renderer",
}
