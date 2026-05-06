# Query-generator validation: Haiku 4.5 vs Sonnet 4.5

Mini-pilot before Block 3B.2 to verify Haiku doesn't regress query quality.
Same prompt, same chunks, deterministic decoding (temperature=0).

## Cost summary

| Model | This pilot (5 queries) | Projected for 800 queries |
|---|---:|---:|
| Haiku 4.5 | $0.0017 | $0.26 |
| Sonnet 4.5 | $0.0057 | $0.92 |
| **Savings using Haiku** | | **$0.66** |

## Per-case comparison

### PROCEDURAL

**Source:** `quantization/finegrained_fp8.md`

**Chunk:**
```
model and reload it with [`~PreTrainedModel.from_pretrained`].

```py
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```
```

**Haiku 4.5** (271→27 tok, $0.0003):

> How do I save a quantized model and load it back later?

**Sonnet 4.5** (271→21 tok, $0.0011):

> how do I save and reload a quantized model in transformers

### COMPARISON

**Source:** `models.md`

**Chunk:**
```
*model* used interchangeably with architecture and checkpoint.

There are two general types of models you can load:

1. A barebones model, like [`AutoModel`] or [`LlamaModel`], that outputs hidden states.
2. A model with a specific *head* attached, like [`AutoModelForCausalLM`] or [`LlamaForCausalLM`], for performing specific tasks.

## Model classes
```

**Haiku 4.5** (286→31 tok, $0.0004):

> Should I use AutoModel or AutoModelForCausalLM for my task?

**Sonnet 4.5** (286→25 tok, $0.0012):

> What's the difference between AutoModel and AutoModelForCausalLM?

### BEST_PRACTICE

**Source:** `model_doc/llava.md`

**Chunk:**
```
Usage tips

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to call `processor.tokenizer.padding_side = "left"` before generating.
```

**Haiku 4.5** (242→30 tok, $0.0003):

> How should I set up padding for batched generation with LLaVA?

**Sonnet 4.5** (242→24 tok, $0.0011):

> what's the recommended padding side for batched generation in LLaVA

### POINTER

**Source:** `add_new_model.md`

**Chunk:**
```
conversion script to copy, adapt, and reuse for your model!

>

> - If you're porting a model from TensorFlow to PyTorch, a good starting point may be the BERT [conversion script](https://github.com/huggingface/transformers/blob/7acfa95afb8194f8f9c1f4d2c6028224dbed35a2/src/transformers/models/bert/modeling_bert.py#L91).
```

**Haiku 4.5** (296→33 tok, $0.0004):

> where can I find a conversion script example for porting TensorFlow models to PyTorch?

**Sonnet 4.5** (296→25 tok, $0.0013):

> Where can I find an example for converting TensorFlow models to PyTorch?

### REFUSAL

**Topic seed:** `AWS Lambda cold start optimization`

**Haiku 4.5** (216→30 tok, $0.0003):

> How do I reduce AWS Lambda cold start time for Python functions?

**Sonnet 4.5** (216→26 tok, $0.0010):

> how do I reduce Lambda cold start times with provisioned concurrency
