# Block 3B.1 pilot — distillation model comparison (prompt v2)

Re-pilot of all four variants under prompt v2, which adds:
- **Adaptive length guidance**: 'match length to what the answer actually requires' instead of an arbitrary sentence cap
- **Anti-skip-prerequisites instruction**: don't drop install steps or basic config to satisfy brevity
- **Anti-URL instruction**: filenames in plain text, no URLs (Opus's v1 Q5 failure mode)
- **Four richer few-shot examples**: a procedural-with-prereqs answer, a comparison answer, a pointer answer, a refusal — calibrating the model to match length to question type

Same input file as v1 (`pilot_input.json`, 6 queries: 5 in-corpus + 1 refusal). Same chunks retrieved across all variants.

## Aggregate stats

| Variant | `[N]` total | Citations on examples | Out-of-range | URLs (target: 0) | Refusals (target: 1) | Mean chars | Pilot cost | Projected 800-example cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Haiku 4.5** | 22 | 5/6 | 0 | 0 | 1 | 264 | $0.014 | ~$2 |
| **Sonnet 4.5** | 26 | 5/6 | 0 | 0 | 1 | 216 | $0.041 | ~$6 |
| **Sonnet 4.5 + thinking** | 24 | 5/6 | 0 | 0 | 1 | 175 | $0.097 | ~$13 |
| **Opus 4.7** | 23 | 5/6 | 0 | 0 | 1 | 254 | $0.289 | ~$39 |

## Per-query side-by-side (prompt v2)

### Q1: How do I save a fine-tuned model and load it back later?

*5 chunks retrieved · expected_refusal=False*

**Haiku 4.5** (4 citations, 355 chars, —):

> Use `model.save_pretrained(path)` to save your fine-tuned model locally [5], then reload it with `AutoModel.from_pretrained(path)` [5]. For example: `model.save_pretrained("/path/to/model")` followed by `model = AutoModelForCausalLM.from_pretrained("/path/to/model")` [3]. This saves the model weights, configuration, and preprocessing class together [5].

**Sonnet 4.5** (4 citations, 332 chars, —):

> Use `save_pretrained()` to save your model locally by passing a directory path [5]. Then reload it with `from_pretrained()` by passing the same directory path [3][5]. For example: `model.save_pretrained("/path/to/save/model")` followed by `model = AutoModelForCausalLM.from_pretrained("/path/to/save/model", device_map="auto")` [3].

**Sonnet 4.5 + thinking** (4 citations, 165 chars, —):

> Use `model.save_pretrained(path)` to save your fine-tuned model to a local directory [3][5]. Reload it later with `from_pretrained(path)` on your model class [3][5].

**Opus 4.7** (2 citations, 280 chars, —):

> Use `save_pretrained()` to save the model (along with its configuration and preprocessing class) locally, then reload it later with `from_pretrained()` [5]. For example: `model.save_pretrained(path)` followed by `AutoModelForCausalLM.from_pretrained(path, device_map="auto")` [3].

### Q2: What's the difference between AutoModel and AutoModelForCausalLM?

*5 chunks retrieved · expected_refusal=False*

**Haiku 4.5** (2 citations, 192 chars, —):

> AutoModel is a base model class without a specific head [3], while AutoModelForCausalLM is a model with a causal language modeling head attached for performing next-token prediction tasks [1].

**Sonnet 4.5** (4 citations, 230 chars, —):

> AutoModel is a barebones model class without a specific head that outputs hidden states [1][3], while AutoModelForCausalLM is a model with a causal language modeling head attached for performing next-token prediction tasks [1][2].

**Sonnet 4.5 + thinking** (2 citations, 165 chars, —):

> AutoModel is a barebones model that outputs hidden states [1], while AutoModelForCausalLM is a model with a specific head attached for performing specific tasks [1].

**Opus 4.7** (4 citations, 217 chars, —):

> AutoModel is a barebones model class without a specific head that outputs hidden states [1][3], while AutoModelForCausalLM is a model with a causal language modeling head attached for performing specific tasks [1][3].

### Q3: How does Flash Attention 2 integrate with transformers models?

*5 chunks retrieved · expected_refusal=False*

**Haiku 4.5** (7 citations, 427 chars, —):

> Flash Attention 2 integrates with transformers models by passing `attn_implementation="flash_attention_2"` to `.from_pretrained()` when loading the model [1][2][3]. First, install the flash-attn package with `pip install -U flash-attn --no-build-isolation` [4][5]. Load the model in half-precision (`torch.float16` or `torch.bfloat16`) to achieve lower memory usage and faster inference with minimal quality degradation [4][5].

**Sonnet 4.5** (10 citations, 373 chars, —):

> Install the flash-attn package first with `pip install -U flash-attn --no-build-isolation` [4][5]. Then pass `attn_implementation="flash_attention_2"` to `.from_pretrained()` when loading the model [1][2][3][4][5]. Load the model in half-precision (`torch.float16` or `torch.bfloat16`) for lower memory usage and faster inference with minimal quality degradation [2][4][5].

**Sonnet 4.5 + thinking** (10 citations, 358 chars, —):

> First install the flash-attn package with `pip install -U flash-attn --no-build-isolation` [4][5]. Then pass `attn_implementation="flash_attention_2"` to `.from_pretrained()` when loading your model [1][2][3][4][5]. Load the model in half-precision (`torch.float16` or `torch.bfloat16`) to get significantly lower memory usage and faster inference [2][4][5].

**Opus 4.7** (9 citations, 662 chars, —):

> Flash Attention 2 is enabled by passing `attn_implementation="flash_attention_2"` to `.from_pretrained()` when loading a model [4][5]. First install the flash-attn package (e.g., `pip install -U flash-attn --no-build-isolation`) [4][5], referring to the original flash-attention repository for installation details [3]. The model should be loaded in half-precision (`torch.float16` or `torch.bfloat16`) to get significantly lower memory usage and faster inference with almost no quality degradation [2][4][5]. For example: `AutoModelForCausalLM.from_pretrained("bigcode/gpt_bigcode-santacoder", dtype=torch.float16, attn_implementation="flash_attention_2")` [1].

### Q4: What's the recommended way to handle padding for batched inference?

*5 chunks retrieved · expected_refusal=False*

**Haiku 4.5** (8 citations, 452 chars, —):

> For batched inference, use `padding_side="left"` as it leads to more accurate results [1][3][4][5]. Set this by calling `processor.tokenizer.padding_side = "left"` before generating [3][4][5]. Alternatively, some models like GraniteMoeHybridForCausalLM support padding-free training which concatenates distinct training examples while still processing inputs as separate batches, potentially accelerating inference by ~2x and reducing memory usage [2].

**Sonnet 4.5** (7 citations, 203 chars, —):

> Use `padding_side="left"` when computing batched generation as it leads to more accurate results [1][3][4][5]. Set this by calling `processor.tokenizer.padding_side = "left"` before generating [3][4][5].

**Sonnet 4.5 + thinking** (7 citations, 201 chars, —):

> Use `padding_side="left"` when computing batched generation as it leads to more accurate results [1][3][4][5]. Set it by calling `processor.tokenizer.padding_side = "left"` before generating [3][4][5].

**Opus 4.7** (7 citations, 202 chars, —):

> Use `padding_side="left"` when computing batched generation, as it leads to more accurate results [1][3][4][5]. Set it by calling `processor.tokenizer.padding_side = "left"` before generating [3][4][5].

### Q5: How do I convert a model from TensorFlow to PyTorch using transformers?

*5 chunks retrieved · expected_refusal=False*

**Haiku 4.5** (1 citations, 120 chars, —):

> Use the BERT conversion script in `modeling_bert.py` as a starting point — copy, adapt, and reuse it for your model [1].

**Sonnet 4.5** (1 citations, 120 chars, —):

> Use the BERT conversion script in `modeling_bert.py` as a starting point — copy, adapt, and reuse it for your model [1].

**Sonnet 4.5 + thinking** (1 citations, 120 chars, —):

> Use the BERT conversion script in `modeling_bert.py` as a starting point — copy, adapt, and reuse it for your model [1].

**Opus 4.7** (1 citations, 121 chars, —):

> A good starting point is the BERT conversion script in `modeling_bert.py` — copy, adapt, and reuse it for your model [1].

### Q6: What's the latest stable Linux kernel version?

*5 chunks retrieved · expected_refusal=True*

**Haiku 4.5** (0 citations, 43 chars, refusal):

> I don't have enough context to answer that.

**Sonnet 4.5** (0 citations, 43 chars, refusal):

> I don't have enough context to answer that.

**Sonnet 4.5 + thinking** (0 citations, 43 chars, refusal):

> I don't have enough context to answer that.

**Opus 4.7** (0 citations, 43 chars, refusal):

> I don't have enough context to answer that.
