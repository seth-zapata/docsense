# Stage 1 query-pool preview (30 sample of 599)

Stratified random sample for hand-review. Eyeball check:

- Are queries realistic? (would a developer actually ask this?)
- Are queries answerable from the retrieved chunks?
- Are retrieved chunks reasonable (not all garbage)?

If a systematic quality issue surfaces, delete the pool and
re-run with adjustments.

## procedural (sample 11 of 221)

**1.** How do I quantize a model on CPU and then load it on GPU?

  - top-3 retrieved doc_ids: `quantization/torchao.md`, `quantization/gptq.md`, `quantization/bitsandbytes.md`
  - seed chunk: `quantization/torchao.md`

**2.** How do I process multiple images at once with GLM-OCR for better performance?

  - top-3 retrieved doc_ids: `model_doc/glm_ocr.md`, `model_doc/glm_ocr.md`, `model_doc/glm_ocr.md`
  - seed chunk: `model_doc/glm_ocr.md`

**3.** How do I get the correct sample rate for saving audio from a Bark model?

  - top-3 retrieved doc_ids: `model_doc/bark.md`, `model_doc/musicgen_melody.md`, `model_doc/bark.md`
  - seed chunk: `model_doc/bark.md`

**4.** How do I set up my environment to upload models to Hugging Face?

  - top-3 retrieved doc_ids: `model_sharing.md`, `model_sharing.md`, `tasks/text-to-speech.md`
  - seed chunk: `tasks/language_modeling.md`

**5.** How do I process audio and text instructions together with Voxtral?

  - top-3 retrieved doc_ids: `model_doc/voxtral.md`, `tasks/any_to_any.md`, `model_doc/voxtral.md`
  - seed chunk: `model_doc/voxtral.md`

**6.** How do I enable bf16 training in my model configuration?

  - top-3 retrieved doc_ids: `model_doc/llama3.md`, `deepspeed.md`, `quantization/torchao.md`
  - seed chunk: `deepspeed.md`

**7.** How do I flatten a dataset and access individual examples from it?

  - top-3 retrieved doc_ids: `tasks/visual_question_answering.md`, `model_doc/pop2piano.md`, `tasks/visual_question_answering.md`
  - seed chunk: `tasks/masked_language_modeling.md`

**8.** How do I visualize which tokens a model can attend to?

  - top-3 retrieved doc_ids: `tasks/token_classification.md`, `tasks/document_question_answering.md`, `model_doc/albert.md`
  - seed chunk: `model_doc/paligemma.md`

**9.** How do I load a quantized model across multiple GPUs automatically?

  - top-3 retrieved doc_ids: `quantization/gptq.md`, `deepspeed.md`, `quantization/gptq.md`
  - seed chunk: `quantization/gptq.md`

**10.** How do I check if a training script supports the max_predict_samples parameter?

  - top-3 retrieved doc_ids: `run_scripts.md`, `run_scripts.md`, `optimizers.md`
  - seed chunk: `run_scripts.md`

**11.** How do I load and set up the Higgs Audio V2 model for multi-speaker generation?

  - top-3 retrieved doc_ids: `model_doc/higgs_audio_v2.md`, `model_doc/higgs_audio_v2.md`, `model_doc/higgs_audio_v2.md`
  - seed chunk: `model_doc/higgs_audio_v2.md`

## comparison (sample 6 of 113)

**1.** When should I use custom models instead of built-in Transformers models?

  - top-3 retrieved doc_ids: `add_new_model.md`, `custom_models.md`, `index.md`
  - seed chunk: `models.md`

**2.** When should I use AutoModelForZeroShotObjectDetection for image tasks?

  - top-3 retrieved doc_ids: `model_doc/auto.md`, `model_doc/auto.md`, `model_doc/auto.md`
  - seed chunk: `model_doc/mm-grounding-dino.md`

**3.** When should I use nested rope_parameters dict vs single rope_parameters?

  - top-3 retrieved doc_ids: `internal/rope_utils.md`, `internal/rope_utils.md`, `internal/rope_utils.md`
  - seed chunk: `internal/rope_utils.md`

**4.** Should I use MobileViT or MobileViTV2 for mobile vision tasks?

  - top-3 retrieved doc_ids: `model_doc/mobilevitv2.md`, `model_doc/mobilevit.md`, `model_doc/mobilevit.md`
  - seed chunk: `model_doc/mobilevitv2.md`

**5.** Should I use Pipeline or AutoModel for Deepseek-VL image-to-text tasks?

  - top-3 retrieved doc_ids: `model_doc/deepseek_vl.md`, `model_doc/deepseek_vl_hybrid.md`, `model_doc/aya_vision.md`
  - seed chunk: `model_doc/deepseek_vl.md`

**6.** How does a video processor differ from an image processor for VLMs?

  - top-3 retrieved doc_ids: `main_classes/video_processor.md`, `video_processors.md`, `main_classes/video_processor.md`
  - seed chunk: `main_classes/video_processor.md`

## best_practice (sample 6 of 113)

**1.** What's the recommended way to prepare images for DepthPro inference?

  - top-3 retrieved doc_ids: `model_doc/depth_pro.md`, `model_doc/depth_pro.md`, `model_doc/depth_pro.md`
  - seed chunk: `model_doc/depth_pro.md`

**2.** How should I handle LLaVA processor warnings about missing attributes?

  - top-3 retrieved doc_ids: `model_doc/llava_next.md`, `model_doc/video_llava.md`, `model_doc/llava_next_video.md`
  - seed chunk: `model_doc/llava_next.md`

**3.** What's the recommended way to prepare images for Mask2Former inference?

  - top-3 retrieved doc_ids: `model_doc/mask2former.md`, `model_doc/mask2former.md`, `model_doc/mask2former.md`
  - seed chunk: `model_doc/mask2former.md`

**4.** What's the recommended way to handle image sizes for LeViT models?

  - top-3 retrieved doc_ids: `model_doc/smolvlm.md`, `model_doc/vit.md`, `model_doc/levit.md`
  - seed chunk: `model_doc/levit.md`

**5.** What's the best way to handle long audio inputs with SeamlessM4Tv2?

  - top-3 retrieved doc_ids: `model_doc/seamless_m4t_v2.md`, `model_doc/seamless_m4t_v2.md`, `model_doc/audioflamingo3.md`
  - seed chunk: `model_doc/seamless_m4t_v2.md`

**6.** How should I configure cache and padding for Flash Attention-2 batched generation?

  - top-3 retrieved doc_ids: `model_doc/mixtral.md`, `model_doc/minimax.md`, `model_doc/olmo_hybrid.md`
  - seed chunk: `model_doc/minimax.md`

## refusal (sample 5 of 92)

**1.** How do I debug XLA compilation errors in JAX HLO lowering?

  - top-3 retrieved doc_ids: `fsdp.md`, `pipeline_webserver.md`, `fsdp.md`
  - seed topic: `JAX HLO compilation and XLA optimization passes`

**2.** How do I check if a training script supports the max_predict_samples parameter?

  - top-3 retrieved doc_ids: `troubleshooting.md`, `fast_tokenizers.md`, `community_integrations/trl.md`
  - seed chunk: `troubleshooting.md`

**3.** How do I implement a custom controller for Kubernetes CRDs with leader election?

  - top-3 retrieved doc_ids: `perf_train_cpu_many.md`, `perf_train_cpu_many.md`, `perf_train_cpu_many.md`
  - seed topic: `Kubernetes operator pattern and Custom Resource Definitions`

**4.** where can I find the best sampling parameters for EXAONE reasoning mode?

  - top-3 retrieved doc_ids: `model_memory_anatomy.md`, `weightconverter.md`, `troubleshooting.md`
  - seed chunk: `model_memory_anatomy.md`

**5.** How do I format messages for a chat pipeline with user and assistant roles?

  - top-3 retrieved doc_ids: `kernel_doc/loading_kernels.md`, `accelerator_selection.md`, `testing.md`
  - seed chunk: `kernel_doc/loading_kernels.md`

## pointer (sample 3 of 60)

**1.** where can I find examples of using apply_chat_template with vectorized output

  - top-3 retrieved doc_ids: `model_doc/llava.md`, `model_doc/llava_next_video.md`, `model_doc/deepseek_v3.md`
  - seed chunk: `model_doc/llava_next_video.md`

**2.** where can I find a notebook showing how to run ViTPose inference?

  - top-3 retrieved doc_ids: `model_doc/vitpose.md`, `model_doc/zoedepth.md`, `model_doc/cvt.md`
  - seed chunk: `model_doc/vitpose.md`

**3.** Where can I find tutorial notebooks to get started with Perceiver?

  - top-3 retrieved doc_ids: `model_doc/perceiver.md`, `model_doc/perceiver.md`, `training.md`
  - seed chunk: `model_doc/perceiver.md`
