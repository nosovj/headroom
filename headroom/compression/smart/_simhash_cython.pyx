# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Optimized simhash and bigram curve using Cython + xxHash for speed.

This module provides Cython-accelerated simhash and bigram curve computation
using xxHash for 6x faster hashing.
"""

import xxhash

def compute_simhash_cython(text):
    """Compute a 64-bit SimHash fingerprint using Cython.

    Args:
        text: Input text.

    Returns:
        64-bit integer fingerprint.
    """
    cdef int v[64]
    cdef int i, j, n
    cdef int fingerprint = 0

    # Initialize vector to zeros
    for i in range(64):
        v[i] = 0

    # Convert to lowercase and get length
    text_lower = text.lower()
    text_len = len(text_lower)

    if text_len == 0:
        return 0

    n = max(1, text_len - 3)

    # Process 4-grams with xxHash (6x faster than MD5)
    for i in range(n):
        # Extract 4-gram
        gram = text_lower[i:i+4]
        # xxHash returns 64-bit directly
        h = xxhash.xxh64(gram, seed=0).intdigest()

        # Update bit vector
        for j in range(64):
            if h & (1 << j):
                v[j] += 1
            else:
                v[j] -= 1

    # Build fingerprint
    fingerprint = 0
    for j in range(64):
        if v[j] > 0:
            fingerprint |= (1 << j)

    return fingerprint


def compute_simhash_batch_cython(texts):
    """Compute simhash for a batch of texts.

    Args:
        texts: List of strings.

    Returns:
        List of 64-bit integer fingerprints.
    """
    cdef int n = len(texts)
    cdef list results = []
    cdef int i

    for i in range(n):
        results.append(compute_simhash_cython(texts[i]))

    return results


def compute_bigram_curve_cython(text):
    """Compute set of word bigrams for a single text using Cython.

    Args:
        text: Input text.

    Returns:
        Set of bigram hashes (as integers for pickling).
    """
    cdef set bigrams = set()
    
    words = text.lower().split()
    n = len(words)
    
    if n == 0:
        return bigrams
    
    if n == 1:
        # Single word - use it as unigram
        bigrams.add((words[0], ''))
    else:
        for i in range(n - 1):
            bigrams.add((words[i], words[i + 1]))
    
    return bigrams


def compute_bigram_curve_batch_cython(texts):
    """Compute bigram sets for a batch of texts.

    Args:
        texts: List of strings.

    Returns:
        List of sets of bigram tuples.
    """
    cdef int n = len(texts)
    cdef list results = []
    cdef int i

    for i in range(n):
        results.append(compute_bigram_curve_cython(texts[i]))

    return results


def compute_simhash_and_bigrams_cython(text):
    """Compute both simhash and bigrams for a text in one pass.

    Returns tuple of (simhash, bigram_set).

    Args:
        text: Input text.

    Returns:
        Tuple of (64-bit simhash, set of bigram tuples).
    """
    cdef int v[64]
    cdef int i, j, n
    cdef int fingerprint = 0
    cdef set bigrams = set()
    
    # Initialize vector to zeros
    for i in range(64):
        v[i] = 0

    # Convert to lowercase and get length
    text_lower = text.lower()
    text_len = len(text_lower)

    if text_len == 0:
        return (0, bigrams)

    n = max(1, text_len - 3)

    # Process 4-grams with xxHash (6x faster than MD5)
    for i in range(n):
        # Extract 4-gram
        gram = text_lower[i:i+4]
        # xxHash returns 64-bit directly
        h = xxhash.xxh64(gram, seed=0).intdigest()

        # Update bit vector
        for j in range(64):
            if h & (1 << j):
                v[j] += 1
            else:
                v[j] -= 1

    # Build fingerprint
    for j in range(64):
        if v[j] > 0:
            fingerprint |= (1 << j)

    # Compute bigrams
    words = text_lower.split()
    num_words = len(words)
    
    if num_words == 0:
        pass
    elif num_words == 1:
        bigrams.add((words[0], ''))
    else:
        for i in range(num_words - 1):
            bigrams.add((words[i], words[i + 1]))

    return (fingerprint, bigrams)
