"""Unit tests for _chunk_text and _concat_wavs in modal_deploy.

These are pure-Python tests. They import modal_deploy (which imports the
modal SDK at module level), so `pip install modal` must succeed in the
test environment. No Modal services are contacted at import time.
"""

import logging

import pytest

from modal_deploy import (
    CHUNK_HARD_CEILING,
    MAX_CHARS_PER_CHUNK,
    _chunk_text,
)


def test_empty_string_returns_single_empty_chunk():
    assert _chunk_text("") == [""]


def test_whitespace_only_returns_single_empty_chunk():
    assert _chunk_text("   \n\t  ") == [""]


def test_single_short_sentence_yields_one_chunk():
    assert _chunk_text("Hello world.") == ["Hello world."]


def test_two_short_sentences_pack_into_one_chunk():
    assert _chunk_text("Hello world. How are you?") == [
        "Hello world. How are you?"
    ]


def test_whitespace_is_normalized_before_packing():
    # Multiple internal spaces / newlines collapse to single spaces.
    assert _chunk_text("Hello.\n\n   World.") == ["Hello. World."]


def test_long_text_splits_into_multiple_chunks_each_within_limits():
    sentence = "This is a moderately long sentence for testing chunking."  # ~56 chars
    text = " ".join([sentence] * 8)  # ~456 chars, 8 sentences
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    for c in chunks:
        # Soft target is 220; hard ceiling is 280. Multi-word chunks
        # must respect the hard ceiling.
        assert len(c) <= CHUNK_HARD_CEILING, c


def test_long_sentence_with_clauses_splits_on_clause_boundaries():
    # Single sentence > max_chars, but full of commas.
    sent = (
        "Today, even though it was raining heavily outside, the entire team gathered, "
        "with notebooks and laptops in hand, in the upstairs meeting room, to carefully "
        "discuss the new release schedule, the open bugs, and the overall testing plan, "
        "before lunch."
    )
    assert len(sent) > MAX_CHARS_PER_CHUNK
    chunks = _chunk_text(sent)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= CHUNK_HARD_CEILING


def test_long_sentence_without_clauses_splits_on_word_boundary_never_mid_word():
    # 60 copies of "word" → ~299 chars, no punctuation at all.
    words = ["word"] * 60
    text = " ".join(words)
    chunks = _chunk_text(text)
    # Every chunk's "tokens" must be intact words.
    for c in chunks:
        for tok in c.split():
            assert tok == "word", f"word split: {tok!r} in chunk {c!r}"


def test_single_word_just_above_max_chars_emitted_as_own_chunk():
    # 250-char single word: > max_chars (220) but <= hard_ceiling (280).
    long_word = "a" * 250
    chunks = _chunk_text(long_word)
    assert chunks == [long_word]


def test_single_word_above_hard_ceiling_emitted_untouched_with_warning(caplog):
    pathological = "a" * 300  # > hard_ceiling (280)
    with caplog.at_level(logging.WARNING, logger="orpheus.chunker"):
        chunks = _chunk_text(pathological)
    assert chunks == [pathological]
    assert any(
        "hard_ceiling" in rec.getMessage() for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


def test_orphan_tail_merges_into_previous_when_within_hard_ceiling():
    # Two long sentences that fit together under hard ceiling, plus a
    # very short tail.
    s_a = "a" * 80 + "."
    s_b = "b" * 80 + "."
    s_c = "End."
    text = " ".join([s_a, s_b, s_c])
    # Force packing with max_chars=165: s_a (81) + " " + s_b (81) = 163,
    # which fits in 165. s_c (4) cannot join (163+1+4=168 > 165). Without
    # merge: ["aaa...", "bbb...", "End."] (s_c is its own chunk, < 30
    # chars). With orphan merge: 163+1+4=168 <= hard_ceiling=280 → merge
    # → one chunk total (per spec §4.1).
    chunks = _chunk_text(text, max_chars=165, hard_ceiling=280)
    # 168 <= hard_ceiling=280, so the orphan-tail merge collapses s_c into
    # the prior chunk → one chunk total.
    assert len(chunks) == 1
    assert chunks[0].endswith("End.")
    assert len(chunks[0]) <= 280


def test_orphan_tail_not_merged_when_would_exceed_hard_ceiling():
    # Force a 2-chunk pack where the tail is short (<30) AND the would-be
    # merged length exceeds hard_ceiling. The merge must be skipped.
    #
    # s_a is exactly max_chars=96 chars. _pack emits it as its own chunk.
    # "End." (4 chars) cannot join (96+1+4=101 > 96) so it lands as an
    # orphan chunk. The merge check: 96+1+4=101 > hard_ceiling=96, so
    # the merge is refused and the orphan tail stays as its own chunk.
    s_a = "a" * 95 + "."  # 96 chars, exactly max_chars
    text = s_a + " End."
    chunks = _chunk_text(text, max_chars=96, hard_ceiling=96)
    assert chunks[-1] == "End."  # orphan stays as its own chunk
    assert len(chunks) >= 2


def test_chunker_is_deterministic():
    text = (
        "First sentence here. Second sentence, with a clause, follows. "
        "Third sentence ends the test."
    )
    assert _chunk_text(text) == _chunk_text(text)


def test_chunker_handles_ethiopic_terminator():
    text = "First sentence። Second sentence።"
    chunks = _chunk_text(text)
    # Both sentences should be preserved verbatim somewhere in the output.
    joined = " ".join(chunks)
    assert "First sentence።" in joined
    assert "Second sentence።" in joined


def test_chunker_respects_max_chars_for_normal_multi_atom_chunks():
    # All atoms small. Every emitted chunk must fit max_chars (no
    # single-atom-overflow path triggered).
    text = " ".join(["short sentence one." for _ in range(40)])
    chunks = _chunk_text(text, max_chars=120, hard_ceiling=160)
    for c in chunks:
        assert len(c) <= 120, c


# --------------------------------------------------------------------------
# _concat_wavs
# --------------------------------------------------------------------------

import numpy as np

from modal_deploy import _concat_wavs


def test_concat_wavs_empty_list_returns_empty_float32_array():
    out = _concat_wavs([])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.size == 0


def test_concat_wavs_single_array_returned_unchanged():
    w = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = _concat_wavs([w])
    np.testing.assert_array_equal(out, w)
    assert out.dtype == np.float32


def test_concat_wavs_two_arrays_inserts_silence_pad():
    w1 = np.ones(1000, dtype=np.float32)
    w2 = np.full(500, 0.5, dtype=np.float32)
    out = _concat_wavs([w1, w2], pad_ms=120)
    expected_pad = int(24000 * 120 / 1000)  # 2880 samples
    assert out.size == 1000 + expected_pad + 500
    np.testing.assert_array_equal(out[:1000], w1)
    np.testing.assert_array_equal(
        out[1000 : 1000 + expected_pad],
        np.zeros(expected_pad, dtype=np.float32),
    )
    np.testing.assert_array_equal(out[1000 + expected_pad :], w2)


def test_concat_wavs_three_arrays_has_two_pads():
    a = np.array([1.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    c = np.array([3.0], dtype=np.float32)
    out = _concat_wavs([a, b, c], pad_ms=10)
    pad_len = int(24000 * 10 / 1000)  # 240
    assert out.size == 3 + 2 * pad_len


def test_concat_wavs_zero_pad_yields_pure_concatenation():
    w1 = np.array([1.0, 2.0], dtype=np.float32)
    w2 = np.array([3.0, 4.0], dtype=np.float32)
    out = _concat_wavs([w1, w2], pad_ms=0)
    np.testing.assert_array_equal(
        out, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )
