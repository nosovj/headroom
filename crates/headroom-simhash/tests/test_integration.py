"""Python integration tests for headroom_simhash Rust extension."""

import pytest

from headroom_simhash import compute_simhash, compute_simhash_batch, count_unique_simhash


class TestComputeSimhash:
    """Tests for compute_simhash single-item function."""

    def test_returns_int(self):
        result = compute_simhash("hello world")
        assert isinstance(result, int)

    def test_positive_value(self):
        result = compute_simhash("hello world")
        assert result >= 0

    def test_deterministic(self):
        text = "hello world testing compression"
        r1 = compute_simhash(text)
        r2 = compute_simhash(text)
        assert r1 == r2

    def test_different_texts_produce_different_hashes(self):
        h1 = compute_simhash("hello")
        h2 = compute_simhash("world")
        # Different texts should produce different hashes (with high probability)
        assert h1 != h2

    def test_empty_string(self):
        result = compute_simhash("")
        assert isinstance(result, int)
        assert result >= 0

    def test_single_char(self):
        result = compute_simhash("a")
        assert isinstance(result, int)
        assert result >= 0

    def test_unicode_text(self):
        result = compute_simhash("こんにちは世界")
        assert isinstance(result, int)
        assert result >= 0

    def test_long_text(self):
        text = "word " * 10000  # 50k chars
        result = compute_simhash(text)
        assert isinstance(result, int)
        assert result >= 0


class TestComputeSimhashBatch:
    """Tests for compute_simhash_batch function."""

    def test_returns_list_of_ints(self):
        texts = ["item1", "item2", "item3"]
        result = compute_simhash_batch(texts)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, int) for x in result)

    def test_batch_size_matches_input(self):
        for n in [1, 10, 50, 100, 500]:
            texts = [f"item_{i}" for i in range(n)]
            result = compute_simhash_batch(texts)
            assert len(result) == n

    def test_deterministic(self):
        texts = [f"test_{i}" for i in range(50)]
        r1 = compute_simhash_batch(texts)
        r2 = compute_simhash_batch(texts)
        assert r1 == r2

    def test_batch_matches_sequential(self):
        texts = ["hello", "world", "testing", "compression"]
        batch_results = compute_simhash_batch(texts)
        sequential_results = [compute_simhash(t) for t in texts]
        assert batch_results == sequential_results

    def test_empty_list(self):
        result = compute_simhash_batch([])
        assert result == []

    def test_large_batch(self):
        texts = [f"item_{i}" for i in range(1000)]
        result = compute_simhash_batch(texts)
        assert len(result) == 1000


class TestCountUniqueSimhash:
    """Tests for count_unique_simhash function."""

    def test_returns_int(self):
        items = ["item1", "item2", "item3"]
        result = count_unique_simhash(items, 3)
        assert isinstance(result, int)

    def test_all_identical_returns_one(self):
        items = ["same", "same", "same"]
        result = count_unique_simhash(items, 3)
        assert result == 1

    def test_all_different_returns_count(self):
        items = [f"unique_{i}" for i in range(50)]
        result = count_unique_simhash(items, 3)
        assert result == 50

    def test_threshold_affects_merging(self):
        items = ["hello world", "hello world", "hello world test"]
        # With threshold 0, they shouldn't merge
        r0 = count_unique_simhash(items, 0)
        # With threshold 10, they might merge
        r10 = count_unique_simhash(items, 10)
        assert isinstance(r0, int)
        assert isinstance(r10, int)

    def test_empty_list(self):
        result = count_unique_simhash([], 3)
        assert result == 0

    def test_threshold_zero(self):
        items = ["a", "b", "c"]
        result = count_unique_simhash(items, 0)
        assert result == 3

    def test_large_input(self):
        items = [f"item_{i}" for i in range(500)]
        result = count_unique_simhash(items, 3)
        assert isinstance(result, int)
        assert result >= 1