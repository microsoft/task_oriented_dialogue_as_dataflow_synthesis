#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import re
from typing import List

import spacy
from spacy.language import Language

from dataflow.core.constants import SpecialStrings


def tokenize_datetime(text: str) -> str:
    """Tokenizes datetime to make it consistent with the seaweed tokens."""
    # 5.10 => 5 . 10
    # 4:00 => 4 : 00
    # 5/7 => 5 / 7
    # 5\7 => 5 \ 7
    # 3-9 => 3 - 9
    text = re.sub(r"(\d)([.:/\\-])(\d)", r"\1 \2 \3", text)

    # 4pm => 4 pm
    text = re.sub(r"(\d+)([a-zA-Z])", r"\1 \2", text)

    # safe guard to avoid multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text


class UtteranceTokenizer:
    """A Spacy-based tokenizer with some heuristics for user utterances."""

    def __init__(self, spacy_model_name: str = "en_core_web_md") -> None:
        self._spacy_nlp: Language = spacy.load(spacy_model_name)

    def tokenize(self, utterance_str: str) -> List[str]:
        """Tokenizes the utterance string and returns a list of tokens.
        """
        if not utterance_str:
            return []

        if utterance_str == SpecialStrings.NULL:
            # do not tokenize the NULL special string
            return [utterance_str]

        tokens: List[str] = sum(
            [
                tokenize_datetime(token.text).split(" ")
                for token in self._spacy_nlp(utterance_str)
            ],
            [],
        )
        return tokens
