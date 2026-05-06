"""Tests for the heuristic chunk-affinity classifier.

Each detector is tested with realistic positive examples (often
distilled from actual chunks in pilot_input.json) and realistic
negative examples to lock the heuristic's behavior. Multi-label cases
verify that overlapping signals produce a union of affinities.
"""

from __future__ import annotations

from docsense.finetuning.chunk_classifier import (
    QuestionType,
    classify_chunk,
    is_best_practice,
    is_comparison,
    is_pointer,
    is_procedural,
)

# --------------------------------------------------------------------
# is_procedural
# --------------------------------------------------------------------


class TestIsProcedural:
    def test_complete_code_block_with_lang_hint(self):
        text = "Example:\n\n```py\nimport torch\nmodel.eval()\n```"
        assert is_procedural(text)

    def test_complete_code_block_no_lang_hint(self):
        text = "Run this:\n\n```\npip install transformers\n```"
        assert is_procedural(text)

    def test_chunk_cut_mid_block_with_lang_hint(self):
        """Chunk boundary may cut mid-block, leaving just the opening
        fence with a language hint. That's still procedural — the
        chunk's content is intended as code."""
        text = "Save your model:\n\n```python\nmodel.save_pretrained("
        assert is_procedural(text)

    def test_orphan_closing_fence_is_not_procedural(self):
        """A chunk that starts after a code-block boundary contains
        only the orphan closing fence — descriptive prose, not code."""
        text = "```\n\nThe above example demonstrates batched generation."
        assert not is_procedural(text)

    def test_repl_prompt(self):
        text = ">>> import torch\n>>> model = AutoModel.from_pretrained('bert-base-uncased')"
        assert is_procedural(text)

    def test_inline_code_only_is_not_procedural(self):
        """Single-backtick inline code (e.g. `save_pretrained`) is
        descriptive, not a procedural example."""
        text = "Use `save_pretrained()` to save your model and `from_pretrained()` to load it."
        assert not is_procedural(text)

    def test_pure_prose(self):
        text = "AutoModel is a class that loads pretrained model weights from the Hub."
        assert not is_procedural(text)

    def test_empty(self):
        assert not is_procedural("")


# --------------------------------------------------------------------
# is_comparison
# --------------------------------------------------------------------


class TestIsComparison:
    def test_two_general_types(self):
        text = "There are two general types of models you can load."
        assert is_comparison(text)

    def test_versus_phrasing(self):
        text = "Choose between AutoModel vs AutoModelForCausalLM based on your task."
        assert is_comparison(text)

    def test_instead_of(self):
        text = "Use AutoModel instead of LlamaModel for cross-architecture compatibility."
        assert is_comparison(text)

    def test_whereas(self):
        text = "AutoModel outputs hidden states whereas AutoModelForCausalLM outputs logits."
        assert is_comparison(text)

    def test_difference(self):
        text = "The key difference between these two classes is the LM head."
        assert is_comparison(text)

    def test_unlike(self):
        text = "Unlike AutoModel, AutoModelForCausalLM has a causal LM head attached."
        assert is_comparison(text)

    def test_compared_to(self):
        text = "Flash Attention is much faster compared to the default implementation."
        assert is_comparison(text)

    def test_pure_descriptive(self):
        text = "AutoModel is a class for loading pretrained models from the Hub."
        assert not is_comparison(text)

    def test_case_insensitive(self):
        text = "AutoModel VS AutoModelForCausalLM."
        assert is_comparison(text)


# --------------------------------------------------------------------
# is_best_practice
# --------------------------------------------------------------------


class TestIsBestPractice:
    def test_we_advise(self):
        text = 'We advise users to use padding_side="left" before generating.'
        assert is_best_practice(text)

    def test_we_recommend(self):
        text = "We recommend setting batch_size to 1 for inference benchmarking."
        assert is_best_practice(text)

    def test_recommended_phrasing(self):
        text = "The recommended approach is to load the model in bf16."
        assert is_best_practice(text)

    def test_best_practice_phrase(self):
        text = "Best practice is to call model.eval() before running inference."
        assert is_best_practice(text)

    def test_you_should(self):
        text = "You should set the pad token before padding sequences."
        assert is_best_practice(text)

    def test_should_be_passive(self):
        text = "The model should be set to eval mode for batched inference."
        assert is_best_practice(text)

    def test_make_sure_to(self):
        text = "Make sure to call processor.tokenizer.padding_side before generating."
        assert is_best_practice(text)

    def test_leads_to_more_accurate(self):
        text = "Padding left leads to more accurate results during batched generation."
        assert is_best_practice(text)

    def test_usage_tips_header(self):
        text = "## Usage tips\n\nLoad the model in bf16 for best performance."
        assert is_best_practice(text)

    def test_bare_tips_header(self):
        text = "Tips\n\n- Set padding_side to left."
        assert is_best_practice(text)

    def test_best_practices_header(self):
        text = "## Best practices\n\nUse bf16 over fp16 for training."
        assert is_best_practice(text)

    def test_lone_should_does_not_match(self):
        """'should' alone is too noisy — descriptive uses ('this should
        not happen', 'the model should produce X') aren't advisory.
        Only 'you should' and 'should be {verb}' are accepted."""
        text = "This should produce a tensor of shape (batch_size, seq_len)."
        assert not is_best_practice(text)

    def test_shouldnt_does_not_match(self):
        """\\b ensures contractions don't trigger a 'should' match."""
        text = "The model shouldn't produce NaN gradients in normal training."
        assert not is_best_practice(text)

    def test_pure_descriptive(self):
        text = "AutoModel is a class that loads pretrained models from the Hub."
        assert not is_best_practice(text)


# --------------------------------------------------------------------
# is_pointer
# --------------------------------------------------------------------


class TestIsPointer:
    def test_good_starting_point(self):
        text = "A good starting point is the BERT conversion script in modeling_bert.py."
        assert is_pointer(text)

    def test_starting_point_alone(self):
        text = "The starting point for any model port is the BERT example."
        assert is_pointer(text)

    def test_refer_to(self):
        text = "Refer to the original repository for installation instructions."
        assert is_pointer(text)

    def test_see_the_repository(self):
        text = "See the flash-attention repository for details on installation."
        assert is_pointer(text)

    def test_for_more_information(self):
        text = "For more information about training, check the trainer docs."
        assert is_pointer(text)

    def test_look_at(self):
        text = "Look at the BERT modeling file for the conversion logic."
        assert is_pointer(text)

    def test_copy_adapt_reuse(self):
        text = "You can copy, adapt, and reuse this conversion script for your own model."
        assert is_pointer(text)

    def test_conversion_script_phrase(self):
        text = "The conversion script handles the TF-to-PyTorch translation automatically."
        assert is_pointer(text)

    def test_pure_descriptive(self):
        text = "AutoModel loads pretrained weights from the Hub."
        assert not is_pointer(text)


# --------------------------------------------------------------------
# classify_chunk — multi-label, real-chunk parity
# --------------------------------------------------------------------


class TestClassifyChunk:
    def test_empty_returns_empty_set(self):
        assert classify_chunk("") == set()

    def test_pure_descriptive_returns_empty_set(self):
        text = "AutoModel is a class for loading pretrained models from the Hub."
        assert classify_chunk(text) == set()

    def test_procedural_only(self):
        text = "Example:\n\n```py\nmodel = AutoModel.from_pretrained('bert-base-uncased')\n```"
        assert classify_chunk(text) == {QuestionType.PROCEDURAL}

    def test_comparison_only(self):
        text = "There are two general types of models: barebones and head-attached."
        assert classify_chunk(text) == {QuestionType.COMPARISON}

    def test_best_practice_only(self):
        text = 'We advise users to set padding_side="left" before generating.'
        assert classify_chunk(text) == {QuestionType.BEST_PRACTICE}

    def test_pointer_only(self):
        text = "A good starting point is the BERT conversion script."
        assert classify_chunk(text) == {QuestionType.POINTER}

    def test_multi_label_procedural_and_best_practice(self):
        """Chunks under a 'Usage tips' header that include code examples
        are common — they appear in both type buckets at sample time."""
        text = (
            "## Usage tips\n\n"
            'We advise using padding_side="left". Example:\n\n'
            "```py\n"
            'processor.tokenizer.padding_side = "left"\n'
            "```"
        )
        assert classify_chunk(text) == {
            QuestionType.PROCEDURAL,
            QuestionType.BEST_PRACTICE,
        }

    def test_multi_label_comparison_and_pointer(self):
        text = (
            "Unlike the default attention implementation, Flash Attention is much "
            "faster. Refer to the original flash-attention repository for details."
        )
        assert classify_chunk(text) == {
            QuestionType.COMPARISON,
            QuestionType.POINTER,
        }

    def test_classify_never_returns_refusal(self):
        """REFUSAL is reserved for refusal-queries seeded separately;
        chunk classification never produces it."""
        text = (
            "Use save_pretrained to save your model, then call from_pretrained later.\n\n"
            "```py\nmodel.save_pretrained('/path')\n```\n\n"
            "We advise using bf16 for inference. Refer to the docs for details."
        )
        assert QuestionType.REFUSAL not in classify_chunk(text)

    # ----------------------------------------------------------------
    # Real-chunk parity: examples paraphrased from pilot_input.json so
    # the heuristics line up with the actual corpus distribution.
    # ----------------------------------------------------------------

    def test_real_chunk_q1_save_load_is_procedural(self):
        # pilot_input.json Q1 chunk 3 (quantization/finegrained_fp8.md)
        text = (
            "model and reload it with [`~PreTrainedModel.from_pretrained`].\n\n"
            "```py\n"
            'quant_path = "/path/to/save/quantized/model"\n'
            "model.save_pretrained(quant_path)\n"
            'model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")\n'
            "```"
        )
        assert QuestionType.PROCEDURAL in classify_chunk(text)

    def test_real_chunk_q2_automodel_is_comparison(self):
        # pilot_input.json Q2 chunk 1 (models.md)
        text = (
            "There are two general types of models you can load:\n\n"
            "1. A barebones model, like [`AutoModel`] or [`LlamaModel`], "
            "that outputs hidden states.\n"
            "2. A model with a specific *head* attached, like [`AutoModelForCausalLM`] "
            "or [`LlamaForCausalLM`], for performing specific tasks."
        )
        assert QuestionType.COMPARISON in classify_chunk(text)

    def test_real_chunk_q4_padding_is_best_practice(self):
        # pilot_input.json Q4 chunk 3 (model_doc/llava.md)
        text = (
            "Usage tips\n\n"
            '- We advise users to use `padding_side="left"` when computing batched '
            "generation as it leads to more accurate results."
        )
        assert QuestionType.BEST_PRACTICE in classify_chunk(text)

    def test_real_chunk_q5_conversion_is_pointer(self):
        # pilot_input.json Q5 chunk 1 (add_new_model.md)
        text = (
            "If you're porting a model from TensorFlow to PyTorch, a good starting "
            "point may be the BERT conversion script."
        )
        assert QuestionType.POINTER in classify_chunk(text)
