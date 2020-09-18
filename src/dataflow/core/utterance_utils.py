#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import re
from typing import List

from dataflow.core.constants import SpecialStrings
from dataflow.core.dialogue import AgentUtterance, UserUtterance
from dataflow.core.utterance_tokenizer import UtteranceTokenizer


def clean_utterance_text(text: str) -> str:
    """Removes line breaking and extra spaces in the user utterance."""
    # sometimes the user utterance contains line breaking and extra spaces
    text = re.sub(r"\s+", " ", text)
    # sometimes the user utterance has leading/ending spaces
    text = text.strip()
    return text


def build_user_utterance(
    text: str, utterance_tokenizer: UtteranceTokenizer
) -> UserUtterance:
    text = clean_utterance_text(text)
    if not text:
        return UserUtterance(
            original_text=SpecialStrings.NULL, tokens=[SpecialStrings.NULL]
        )
    return UserUtterance(original_text=text, tokens=utterance_tokenizer.tokenize(text))


def build_agent_utterance(
    text: str, utterance_tokenizer: UtteranceTokenizer, described_entities: List[str]
) -> AgentUtterance:
    text = clean_utterance_text(text)
    if not text:
        return AgentUtterance(
            original_text=SpecialStrings.NULL,
            tokens=[SpecialStrings.NULL],
            described_entities=described_entities,
        )
    return AgentUtterance(
        original_text=text,
        tokens=utterance_tokenizer.tokenize(text),
        described_entities=described_entities,
    )
