# ComfyUI-LTXDirectorSimple

Simplified all-in-one nodes for LTX Video generation with Director workflow.

Replaces the chain of: LTXDirector → ConditioningZeroOut → LTXVConditioning → LTXDirectorGuide → ConcatAV → Guider → Scheduler → Sampler → SeparateAV → CropGuides → Upsampler → DirGuide2 → ConcatAV → Sampler → SeparateAV → CropGuides → Decode

With just: LTXDirector → LTXDirectorSampler (one node for the entire sampling pipeline)
