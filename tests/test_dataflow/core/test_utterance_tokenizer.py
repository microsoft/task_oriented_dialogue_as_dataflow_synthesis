#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from dataflow.core.utterance_tokenizer import UtteranceTokenizer, tokenize_datetime


def test_tokenize_datetime():
    data = [
        ("5.10", "5 . 10"),
        ("4:00", "4 : 00"),
        ("5/7", "5 / 7"),
        ("5\\7", "5 \\ 7"),
        ("3-9", "3 - 9"),
        ("3pm", "3 pm"),
    ]
    for text, expected in data:
        assert tokenize_datetime(text) == expected


def test_tokenize_utterance():
    utterance_tokenizer = UtteranceTokenizer()

    data = [
        (
            "Reschedule meeting with Barack Obama to 5/30/2019 at 3:00pm",
            [
                "Reschedule",
                "meeting",
                "with",
                "Barack",
                "Obama",
                "to",
                "5",
                "/",
                "30",
                "/",
                "2019",
                "at",
                "3",
                ":",
                "00",
                "pm",
            ],
        ),
        (
            "Can you also add icecream birthday tomorrow at 6PM?",
            [
                "Can",
                "you",
                "also",
                "add",
                "icecream",
                "birthday",
                "tomorrow",
                "at",
                "6",
                "PM",
                "?",
            ],
        ),
    ]
    for text, expected in data:
        assert utterance_tokenizer.tokenize(text) == expected
