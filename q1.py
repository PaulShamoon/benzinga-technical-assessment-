from collections import (
    Counter,
    defaultdict
)

Pair = tuple[str, str]


def _adjacent_pairs(symbols: tuple[str, ...]) -> list[Pair]:
    """Helper method to find every neighbouring pair in a word, one entry per occurrence

    Args:
        symbols (tuple[str, ...]): The word as a tuple of symbols

    Returns:
        list[Pair]: Adjacent (left, right) pairs in order of appearance
    """
    return list(zip(symbols, symbols[1:]))


def _merge_pair(
    symbols: tuple[str, ...],
    pair: Pair
) -> tuple[str, ...]:
    """Helper method to merge every occurrence of 'pair' in a word, scans left to right

    Args:
        symbols (tuple[str, ...]): The word as a tuple of symbols
        pair (Pair): The (left, right) pair to merge into a single symbol

    Returns:
        tuple[str, ...]: The word with every occurrence of 'pair' merged
    """
    merged_symbol = pair[0] + pair[1]
    result = []
    i = 0

    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(merged_symbol)
            i += 2
        else:
            result.append(symbols[i])
            i += 1

    return tuple(result)


def _add_word(
    word_id: int,
    word: tuple[str, ...],
    freq: int,
    pair_counts: Counter[Pair],
    pair_to_words: defaultdict[Pair, set[int]]
) -> None:
    """Helper method to add a word's pairs to the counts and the pair_to_words idx

    Args:
        word_id (int): Index of the word in 'words'
        word (tuple[str, ...]): The word as a tuple of symbols
        freq (int): The word's frequency
        pair_counts (Counter[Pair]): Total weighted count of each pair
        pair_to_words (defaultdict[Pair, set[int]]): For each pair, the ids of words containing it
    """
    for pair in _adjacent_pairs(word):
        pair_counts[pair] += freq
        pair_to_words[pair].add(word_id)


def _remove_word(
    word_id: int,
    word: tuple[str, ...],
    freq: int,
    pair_counts: Counter[Pair],
    pair_to_words: defaultdict[Pair, set[int]]
) -> None:
    """Helper method to remove a word's pairs from the counts and the pair_to_words idx

    Args:
        word_id (int): Index of the word in 'words'
        word (tuple[str, ...]): The word as a tuple of symbols
        freq (int): The word's frequency
        pair_counts (Counter[Pair]): Total weighted count of each pair
        pair_to_words (defaultdict[Pair, set[int]]): For each pair, the ids of words containing it
    """
    for pair in _adjacent_pairs(word):
        pair_counts[pair] -= freq

        if pair_counts[pair] == 0:
            del pair_counts[pair]

        pair_to_words[pair].discard(word_id)


# Part A

def train(
    corpus: list[tuple[str, int]],
    num_merges: int
) -> list[Pair]:
    """Learn up to 'num_merges' BPE merge rules from (word, frequency) pairs

    Args:
        corpus (list[tuple[str, int]]): (word, frequency) pairs
        num_merges (int): Max number of merge rules to learn

    Returns:
        list[Pair]: The merge rules in the order they were learned
    """
    words = [tuple(word) for word, _ in corpus]
    freqs = [freq for _, freq in corpus]

    pair_counts: Counter[Pair] = Counter()
    pair_to_words: defaultdict[Pair, set[int]] = defaultdict(set)

    for word_id, word in enumerate(words):
        _add_word(word_id, word, freqs[word_id], pair_counts, pair_to_words)

    merges: list[Pair] = []
    for _ in range(num_merges):
        if not pair_counts:
            break

        best = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
        if pair_counts[best] <= 1:
            break

        for word_id in list(pair_to_words[best]):
            freq = freqs[word_id]
            _remove_word(word_id, words[word_id], freq, pair_counts, pair_to_words)
            words[word_id] = _merge_pair(words[word_id], best)
            _add_word(word_id, words[word_id], freq, pair_counts, pair_to_words)

        merges.append(best)

    return merges


# Part B

def encode(
    word: str,
    merges: list[Pair]
) -> list[str]:
    """Split 'word' into subword tokens by applying merges in the learned order

    Args:
        word (str): The word to tokenize
        merges (list[Pair]): Merge rules from 'train in the learned order

    Returns:
        list[str]: The subword tokens for 'word'
    """
    symbols = tuple(word)

    for pair in merges:
        symbols = _merge_pair(symbols, pair)

    return list(symbols)


if __name__ == "__main__":
    example_words = [("low", 5), ("lower", 2), ("newest", 6), ("widest", 3)]
    learned_merges = train(example_words, 4)

    print(learned_merges)
    print(encode("lowest", learned_merges))
    print(encode("newer", learned_merges)) 

    # Other test cases
   
    # example_words = [("aaa", 3)]
    # learned_merges = train(example_words, 5)
    # print(learned_merges)

    # example_words = [("ab", 1), ("cd", 1)]
    # learned_merges = train(example_words, 5)
    # print(learned_merges)

    # example_words = [("ab", 2), ("ac", 2)]
    # learned_merges = train(example_words, 5)
    # print(learned_merges)
