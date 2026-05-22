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
            seed, steps, 1.0, 1.0, sampler_name, scheduler
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
                seed, upscale_steps, upscale_denoise, 1.0, sampler_name, scheduler
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
        """Apply LTXDirectorGuide: encode guide images into latent at their frame positions."""
        from comfy_extras.nodes_lt import LTXVAddGuide, get_keyframe_idxs

        latent_image = video_latent["samples"].clone()
        noise_mask = get_noise_mask(video_latent).clone()
        _, _, latent_length, latent_height, latent_width = latent_image.shape

        if not guide_data or len(guide_data) == 0 or strength <= 0:
            return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}

        scale_factors = vae.downscale_index_formula

        for guide in guide_data:
            image = guide["image"]
            frame_idx = guide.get("frame_idx", 0)
            guide_str = guide.get("strength", strength)

            if guide_str <= 0:
                continue

            # Encode image through VAE (same as LTXVAddGuide.encode)
            _, encoded = LTXVAddGuide.encode(vae, latent_width, latent_height, image, scale_factors)

            # Calculate latent frame index
            frame_idx_resolved, latent_idx = LTXVAddGuide.get_latent_index(
                positive, latent_length, image.shape[0], frame_idx, scale_factors
            )

            # Append keyframe to latent and conditioning
            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx_resolved, latent_image, noise_mask,
                encoded, guide_str, scale_factors
            )

            latent_length = latent_image.shape[2]

        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}

    def _sample_pass(self, model, positive, negative, video_latent, audio_latent,
                     seed, steps, denoise, cfg, sampler_name, scheduler_name):
        """Run one sampling pass: concat AV → sample → separate AV."""
        # Concat video + audio
        av_latent = self._concat_av(video_latent, audio_latent)

        # Create noise
        noise = comfy.sample.prepare_noise(av_latent["samples"], seed)

        # Create sampler and sigmas
        real_model = model.model if hasattr(model, 'model') else model
        sampler = comfy.samplers.sampler_object(sampler_name)

        sigmas = comfy.samplers.calculate_sigmas(
            real_model.model_sampling, scheduler_name, steps
        ).cpu()

        # Apply denoise (trim sigmas)
        if denoise < 1.0:
            total = len(sigmas) - 1
            start_step = max(0, int(total * (1.0 - denoise)))
            sigmas = sigmas[start_step:]

        # Build guider (CFGGuider equivalent)
        noise_mask = av_latent.get("noise_mask", None)

        # Use ComfyUI's sample_custom for maximum compatibility
        latent_image = av_latent["samples"]

        disable_pbar = False
        callback = None

        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas,
            positive, negative, latent_image,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
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
            try:
                masks = av_latent["noise_mask"].unbind()
                video["noise_mask"] = masks[0]
                audio["noise_mask"] = masks[1]
            except:
                pass

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

        Uses VAE per-channel statistics to un-normalize before upscaling,
        then re-normalizes after. This matches ComfyUI's LTXVLatentUpsampler exactly.
        """
        latents = video_latent["samples"]

        # Un-normalize using VAE statistics, upscale, re-normalize
        latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
        upscaled = upscale_model(latents)
        upscaled = vae.first_stage_model.per_channel_statistics.normalize(upscaled)

        result = video_latent.copy()
        result["samples"] = upscaled

        # Upscale noise mask to match new spatial dimensions
        if "noise_mask" in video_latent:
            mask = video_latent["noise_mask"]
            b, mc, t, h, w = mask.shape
            new_h, new_w = upscaled.shape[-2], upscaled.shape[-1]
            reshaped_mask = mask.permute(0, 2, 1, 3, 4).reshape(b * t, mc, h, w)
            upscaled_mask = comfy.utils.common_upscale(
                reshaped_mask, new_w, new_h, "nearest", "center"
            )
            result["noise_mask"] = upscaled_mask.reshape(b, t, mc, new_h, new_w).permute(0, 2, 1, 3, 4)

        return result


NODE_CLASS_MAPPINGS = {
    "LTXShotRenderer": LTXShotRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXShotRenderer": "LTX Shot Renderer",
}
