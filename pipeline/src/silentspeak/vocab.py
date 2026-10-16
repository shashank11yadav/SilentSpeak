"""The GRID vocabulary.

Every GRID sentence follows one fixed grammar,
``command colour preposition letter digit adverb`` (for example "bin blue at f
two now"), so each word belongs to exactly one slot. The demo uses those slots
as categories: picking a category narrows the choice from 51 words to 4-25.
"""

from __future__ import annotations

from typing import Final

CATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    "command": ("bin", "lay", "place", "set"),
    "colour": ("blue", "green", "red", "white"),
    "preposition": ("at", "by", "in", "with"),
    "letter": tuple("abcdefghijklmnopqrstuvxyz"),  # GRID has no "w"
    "digit": ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    "adverb": ("again", "now", "please", "soon"),
}

ALL_WORDS: Final[tuple[str, ...]] = tuple(w for words in CATEGORIES.values() for w in words)

WORD_CATEGORY: Final[dict[str, str]] = {
    word: category for category, words in CATEGORIES.items() for word in words
}

# Alignment tokens that are not words.
NON_WORDS: Final[frozenset[str]] = frozenset({"sil", "sp"})
