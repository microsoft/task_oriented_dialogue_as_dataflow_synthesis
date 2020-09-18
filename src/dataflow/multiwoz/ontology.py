#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
TRADE_SLOT_NAMES_FOR_DOMAIN = {
    "attraction": ["area", "name", "type"],
    "hotel": [
        "area",
        "book day",
        "book people",
        "book stay",
        "internet",
        "name",
        "parking",
        "pricerange",
        "stars",
        "type",
    ],
    "restaurant": [
        "area",
        "book day",
        "book people",
        "book time",
        "food",
        "name",
        "pricerange",
    ],
    "taxi": ["arriveby", "departure", "destination", "leaveat"],
    "train": ["arriveby", "book people", "day", "departure", "destination", "leaveat"],
    "bus": ["day", "departure", "destination", "leaveat"],
    "hospital": ["department"],
}

# The slot names used in dataflow.
# NOTE: We cannot use the original TRADE_SLOT_NAMES_FOR_DOMAIN because space is not allowed in dataflow slot names.
DATAFLOW_SLOT_NAMES_FOR_DOMAIN = {
    "attraction": ["area", "name", "type"],
    "hotel": [
        "area",
        "book-day",
        "book-people",
        "book-stay",
        "internet",
        "name",
        "parking",
        "pricerange",
        "stars",
        "type",
    ],
    "restaurant": [
        "area",
        "book-day",
        "book-people",
        "book-time",
        "food",
        "name",
        "pricerange",
    ],
    "taxi": ["arriveby", "departure", "destination", "leaveat"],
    "train": ["arriveby", "book-people", "day", "departure", "destination", "leaveat"],
    "bus": ["day", "departure", "destination", "leaveat"],
    "hospital": ["department"],
}
