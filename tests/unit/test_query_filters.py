"""Unit tests for query-pool quality filters.

The dedupe and contamination filters need an embedder, so tests use
a deterministic mock that maps known texts to pre-specified vectors.
This keeps similarity math predictable and avoids loading a real
sentence-transformers model in unit tests.
"""

from __future__ import annotations

import numpy as np

from docsense.finetuning.chunk_classifier import QuestionType
from docsense.finetuning.query_filters import (
    FilterReport,
    filter_by_length,
    filter_by_type_shape,
    filter_duplicates,
    filter_eval_contamination,
)
from docsense.finetuning.query_generation import GeneratedQuery


def _make_query(
    text: str,
    qt: QuestionType = QuestionType.PROCEDURAL,
) -> GeneratedQuery:
    return GeneratedQuery(query=text, question_type=qt)


def _mock_embedder(text_to_vec: dict[str, np.ndarray]):
    """Return an embedder that looks up texts in the provided dict.

    Unknown texts map to a zero vector (dot product = 0, similarity
    = 0). Test failures show clearly when a test sends a text we
    forgot to map.
    """
    default = np.zeros(3)

    def encode(texts: list[str]) -> np.ndarray:
        return np.array([text_to_vec.get(t, default) for t in texts])

    return encode


# --------------------------------------------------------------------
# filter_by_length
# --------------------------------------------------------------------


class TestFilterByLength:
    def test_keeps_queries_meeting_min_words(self):
        queries = [
            _make_query("how do I save a model"),  # 6 words
            _make_query("what is the recommended padding side"),  # 6 words
        ]
        report = filter_by_length(queries, min_words=5)
        assert report.kept_count == 2
        assert report.dropped_count == 0

    def test_drops_queries_below_min_words(self):
        queries = [
            _make_query("what is BERT"),  # 3 words — fails
            _make_query("how do I save a fine-tuned model"),  # 7 words — passes
        ]
        report = filter_by_length(queries, min_words=5)
        assert report.kept_count == 1
        assert report.dropped_count == 1
        assert report.dropped[0][0].query == "what is BERT"
        assert "length" in report.dropped[0][1]

    def test_min_words_is_inclusive_floor(self):
        """A query with exactly min_words words is kept."""
        queries = [_make_query("save model load again now")]  # 5 words
        report = filter_by_length(queries, min_words=5)
        assert report.kept_count == 1

    def test_default_min_words_is_5(self):
        queries = [_make_query("one two three four")]  # 4 words
        report = filter_by_length(queries)
        assert report.dropped_count == 1

    def test_empty_input(self):
        report = filter_by_length([])
        assert report.kept_count == 0
        assert report.dropped_count == 0


# --------------------------------------------------------------------
# filter_by_type_shape
# --------------------------------------------------------------------


class TestFilterByTypeShape:
    """Drops queries whose phrasing doesn't match their assigned type.

    Each test uses real query phrasings observed in the first Stage 1 run
    (or close paraphrases) so the patterns are calibrated to actual
    Haiku output, not synthetic examples.
    """

    def test_procedural_starting_with_how_is_kept(self):
        q = _make_query("how do I save a fine-tuned model", QuestionType.PROCEDURAL)
        report = filter_by_type_shape([q])
        assert report.kept_count == 1

    def test_procedural_not_starting_with_how_is_dropped(self):
        """A query like 'What is BERT?' in the procedural bucket
        doesn't match the procedural shape — drop it."""
        q = _make_query("what is BERT used for in transformers", QuestionType.PROCEDURAL)
        report = filter_by_type_shape([q])
        assert report.dropped_count == 1
        assert "type_mismatch_procedural" in report.dropped[0][1]

    def test_comparison_with_difference_vocabulary_kept(self):
        for text in [
            "what's the difference between AutoModel and AutoModelForCausalLM",
            "how is image classification different from audio classification",
            "AutoModel vs AutoModelForCausalLM for embeddings",
            "should I use int4 or int8 quantization for accuracy",
        ]:
            q = _make_query(text, QuestionType.COMPARISON)
            assert filter_by_type_shape([q]).kept_count == 1, text

    def test_comparison_without_comparison_vocab_dropped(self):
        """Real example from the first Stage 1 run that surfaced the
        type-drift issue: 'How do I load WikiText-2 and prepare it
        for perplexity evaluation' in the comparison bucket."""
        q = _make_query(
            "how do I load WikiText-2 and prepare it for perplexity evaluation",
            QuestionType.COMPARISON,
        )
        report = filter_by_type_shape([q])
        assert report.dropped_count == 1

    def test_best_practice_with_advisory_vocab_kept(self):
        for text in [
            "what's the recommended way to handle padding for batched inference",
            "what's the best way to load a vision model for image-to-text tasks",
            "should I use a tokenizer with CANINE for batched inference",
            "how should I convert MusicGen checkpoints to transformers format",
        ]:
            q = _make_query(text, QuestionType.BEST_PRACTICE)
            assert filter_by_type_shape([q]).kept_count == 1, text

    def test_best_practice_without_advisory_vocab_dropped(self):
        """Real example: 'How do I explicitly enable SDPA attention
        when loading a model' was tagged best_practice but is procedural."""
        q = _make_query(
            "how do I explicitly enable SDPA attention when loading a model",
            QuestionType.BEST_PRACTICE,
        )
        report = filter_by_type_shape([q])
        assert report.dropped_count == 1

    def test_pointer_with_where_kept(self):
        for text in [
            "where can I find AutoRound and what's its relationship to Intel Neural Compressor",
            "where do I configure ZeRO optimization stages in transformers",
            "where is the BERT conversion script located",
        ]:
            q = _make_query(text, QuestionType.POINTER)
            assert filter_by_type_shape([q]).kept_count == 1, text

    def test_pointer_without_where_dropped(self):
        """Real examples: 'How do I quantize a model to int4 using
        bitsandbytes' and 'How do I load a T5 model' were tagged
        pointer but are procedural in shape."""
        for text in [
            "how do I quantize a model to int4 using bitsandbytes",
            "how do I load a T5 model for sequence-to-sequence training tasks",
        ]:
            q = _make_query(text, QuestionType.POINTER)
            report = filter_by_type_shape([q])
            assert report.dropped_count == 1, text

    def test_refusal_is_exempt(self):
        """Refusals are defined by chunk-mismatch (off-corpus topic OR
        mismatched chunks), not phrasing. Any shape is accepted."""
        for text in [
            "how do I tune AWS Lambda cold starts",  # how-shape
            "what's the recommended postgres VACUUM cadence",  # advisory
            "where can I find the Linux kernel scheduler source",  # where
            "should I use Raft or Paxos for consensus",  # comparison-shape
        ]:
            q = _make_query(text, QuestionType.REFUSAL)
            assert filter_by_type_shape([q]).kept_count == 1, text

    def test_empty_input(self):
        assert filter_by_type_shape([]).kept_count == 0

    def test_mixed_types_filtered_independently(self):
        """A batch of mixed-type queries: each is checked against ITS
        own type's shape, not against a single global pattern."""
        queries = [
            _make_query("how do I X", QuestionType.PROCEDURAL),  # kept
            _make_query("what is X", QuestionType.PROCEDURAL),  # dropped
            _make_query("difference between X and Y", QuestionType.COMPARISON),  # kept
            _make_query("how do I X", QuestionType.COMPARISON),  # dropped (no comparison vocab)
        ]
        report = filter_by_type_shape(queries)
        assert report.kept_count == 2
        assert report.dropped_count == 2


# --------------------------------------------------------------------
# filter_duplicates — embedding-based, type-stratified
# --------------------------------------------------------------------


class TestFilterDuplicates:
    def test_drops_near_duplicate_within_same_type(self):
        v_save = np.array([1.0, 0.0, 0.0])
        v_other = np.array([0.0, 1.0, 0.0])
        embedder = _mock_embedder(
            {
                "how do I save a model?": v_save,
                "how can I save my model?": v_save,  # identical embedding
                "what is BERT used for?": v_other,
            }
        )
        queries = [
            _make_query("how do I save a model?"),
            _make_query("how can I save my model?"),
            _make_query("what is BERT used for?"),
        ]
        report = filter_duplicates(queries, embedder, threshold=0.9)
        assert report.kept_count == 2
        assert report.dropped_count == 1
        assert report.kept[0].query == "how do I save a model?"
        assert report.dropped[0][0].query == "how can I save my model?"
        assert "duplicate_within_procedural" in report.dropped[0][1]

    def test_keeps_cross_type_similar_queries(self):
        """A procedural query and a comparison query that embed
        similarly are NOT deduped — they answer different question
        shapes, so the training signals are distinct."""
        v_save = np.array([1.0, 0.0, 0.0])
        embedder = _mock_embedder(
            {
                "how do I save a model?": v_save,
                "save model vs save_pretrained difference?": v_save,
            }
        )
        queries = [
            _make_query("how do I save a model?", QuestionType.PROCEDURAL),
            _make_query(
                "save model vs save_pretrained difference?",
                QuestionType.COMPARISON,
            ),
        ]
        report = filter_duplicates(queries, embedder, threshold=0.9)
        assert report.kept_count == 2
        assert report.dropped_count == 0

    def test_keeps_dissimilar_queries(self):
        embedder = _mock_embedder(
            {
                "how do I save a model?": np.array([1.0, 0.0, 0.0]),
                "what is the recommended padding side?": np.array([0.0, 1.0, 0.0]),
            }
        )
        queries = [
            _make_query("how do I save a model?"),
            _make_query("what is the recommended padding side?"),
        ]
        report = filter_duplicates(queries, embedder, threshold=0.9)
        assert report.kept_count == 2

    def test_threshold_calibration(self):
        """Threshold=0.9 keeps a borderline pair that threshold=0.5
        would dedupe — pin the threshold's effect."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.7, 0.7])  # cos with v1 ≈ 0.707 after normalization
        embedder = _mock_embedder(
            {
                "first query that has enough words": v1,
                "second query that has enough words": v2,
            }
        )
        queries = [
            _make_query("first query that has enough words"),
            _make_query("second query that has enough words"),
        ]
        # 0.707 is below 0.9 → both kept
        strict = filter_duplicates(queries, embedder, threshold=0.9)
        assert strict.kept_count == 2
        # 0.707 is above 0.5 → second is dropped
        loose = filter_duplicates(queries, embedder, threshold=0.5)
        assert loose.kept_count == 1
        assert loose.dropped_count == 1

    def test_chains_of_duplicates_all_dropped_except_first(self):
        """If A ~ B ~ C all embed identically, only A is kept."""
        v = np.array([1.0, 0.0, 0.0])
        embedder = _mock_embedder(
            {
                "query alpha sample text": v,
                "query beta sample text": v,
                "query gamma sample text": v,
            }
        )
        queries = [
            _make_query("query alpha sample text"),
            _make_query("query beta sample text"),
            _make_query("query gamma sample text"),
        ]
        report = filter_duplicates(queries, embedder, threshold=0.9)
        assert report.kept_count == 1
        assert report.kept[0].query == "query alpha sample text"
        assert report.dropped_count == 2

    def test_empty_input(self):
        embedder = _mock_embedder({})
        report = filter_duplicates([], embedder)
        assert report.kept_count == 0

    def test_single_query_no_dedupe(self):
        """A single query passes through without invoking the embedder
        comparison loop."""
        embedder = _mock_embedder({"only query here passes through": np.array([1.0, 0.0, 0.0])})
        queries = [_make_query("only query here passes through")]
        report = filter_duplicates(queries, embedder)
        assert report.kept_count == 1


# --------------------------------------------------------------------
# filter_eval_contamination — unstratified
# --------------------------------------------------------------------


class TestFilterEvalContamination:
    def test_drops_queries_similar_to_eval(self):
        v_save = np.array([1.0, 0.0, 0.0])
        v_other = np.array([0.0, 1.0, 0.0])
        embedder = _mock_embedder(
            {
                "how do I save a fine-tuned model?": v_save,  # train
                "what is the recommended padding?": v_other,  # train
                "how do I save my model?": v_save,  # eval — collides with train[0]
            }
        )
        train = [
            _make_query("how do I save a fine-tuned model?"),
            _make_query("what is the recommended padding?"),
        ]
        evals = ["how do I save my model?"]
        report = filter_eval_contamination(train, evals, embedder, threshold=0.9)
        assert report.kept_count == 1
        assert report.kept[0].query == "what is the recommended padding?"
        assert report.dropped_count == 1
        assert "eval_contamination" in report.dropped[0][1]

    def test_not_type_stratified(self):
        """Cross-type overlap with eval queries IS contamination — the
        eval comparison gets corrupted regardless of question type."""
        v_shared = np.array([1.0, 0.0, 0.0])
        embedder = _mock_embedder(
            {
                "what's the difference between save and load?": v_shared,
                "how do I save a model?": v_shared,
            }
        )
        train = [
            _make_query(
                "what's the difference between save and load?",
                QuestionType.COMPARISON,
            ),
        ]
        evals = ["how do I save a model?"]  # would be PROCEDURAL eval
        report = filter_eval_contamination(train, evals, embedder, threshold=0.9)
        assert report.dropped_count == 1

    def test_keeps_dissimilar_queries(self):
        embedder = _mock_embedder(
            {
                "train query about transformers internals": np.array([1.0, 0.0, 0.0]),
                "eval query about kernels and graphs": np.array([0.0, 1.0, 0.0]),
            }
        )
        train = [_make_query("train query about transformers internals")]
        evals = ["eval query about kernels and graphs"]
        report = filter_eval_contamination(train, evals, embedder, threshold=0.7)
        assert report.kept_count == 1
        assert report.dropped_count == 0

    def test_empty_eval_set_keeps_everything(self):
        """No eval set to check against → no contamination possible."""
        embedder = _mock_embedder({})
        queries = [_make_query("any query at all passes through")]
        report = filter_eval_contamination(queries, [], embedder)
        assert report.kept_count == 1
        assert report.dropped_count == 0

    def test_empty_train_set(self):
        embedder = _mock_embedder({})
        report = filter_eval_contamination([], ["eval query here"], embedder)
        assert report.kept_count == 0
        assert report.dropped_count == 0

    def test_default_threshold_07(self):
        """Default threshold 0.7 catches loose paraphrases — pin the
        default so it doesn't silently drift."""
        # cos ≈ 0.707 (from v1=(1,0), v2=(1,1) normalized) → just above 0.7
        embedder = _mock_embedder(
            {
                "train query for default threshold test": np.array([1.0, 0.0]),
                "eval query for default threshold test": np.array([1.0, 1.0]),
            }
        )
        train = [_make_query("train query for default threshold test")]
        evals = ["eval query for default threshold test"]
        report = filter_eval_contamination(train, evals, embedder)
        assert report.dropped_count == 1


# --------------------------------------------------------------------
# FilterReport
# --------------------------------------------------------------------


class TestFilterReport:
    def test_reason_summary_aggregates_drops_by_reason(self):
        q1 = _make_query("query one passes through fine")
        q2 = _make_query("two short")
        q3 = _make_query("also too short")
        q4 = _make_query("contaminated")
        report = FilterReport()
        report.kept = [q1]
        report.dropped = [
            (q2, "length<5_words"),
            (q3, "length<5_words"),
            (q4, "eval_contamination_sim=0.812"),
        ]
        summary = report.reason_summary()
        assert summary["length<5_words"] == 2
        assert summary["eval_contamination_sim=0.812"] == 1

    def test_total_counts_kept_plus_dropped(self):
        report = FilterReport()
        report.kept = [_make_query("kept query passes through fine")]
        report.dropped = [(_make_query("short"), "reason")]
        assert report.total == 2

    def test_empty_report_summary(self):
        report = FilterReport()
        assert report.total == 0
        assert report.reason_summary() == {}
