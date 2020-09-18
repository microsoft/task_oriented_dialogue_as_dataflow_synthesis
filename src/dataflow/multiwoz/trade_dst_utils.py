#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import re
from typing import Any, Dict, List, Tuple, Union

_Slots = List[List[str]]
_Act = str
BeliefState = List[Dict[str, Union[_Slots, _Act]]]

# From https://github.com/jasonwu0731/trade-dst/blob/master/utils/utils_multiWOZ_DST.py.
_EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def concatenate_system_and_user_transcript(turn: Dict[str, Any]) -> str:
    """Returns the concatenated (agentUtterance, userUtterance)."""
    return "{} {}".format(turn["system_transcript"].strip(), turn["transcript"].strip())


def get_domain_and_slot_name(slot_fullname: str) -> Tuple[str, str]:
    """Returns the domain in a slot fullname."""
    units = slot_fullname.split("-")
    return units[0], "-".join(units[1:])


def trade_normalize_slot_name(name: str) -> str:
    """Normalizes the slot name as in TRADE.

    Extracted from get_slot_information in https://github.com/jasonwu0731/trade-dst/blob/master/utils/utils_multiWOZ_DST.py.
    """
    if "book" not in name:
        return name.replace(" ", "").lower()
    return name.lower()


def fix_general_label_error(trade_belief_state_labels: BeliefState) -> Dict[str, str]:
    """Fixes some label errors in MultiWoZ.

    Adapted from https://github.com/jasonwu0731/trade-dst/blob/master/utils/fix_label.py.
    - Removed the "type" argument, which is always False.
    - "ALL_SLOTS" are hard-coded in the method now

    NOTE: the trade_belief_state_labels may not share the same slot name as dataflow.

    Args:
        trade_belief_state_labels: TRADE processed original belief state
    Returns:
        the flatten belief dictionary with corrected slot values
    """
    label_dict: Dict[str, str] = {
        l["slots"][0][0]: l["slots"][0][1] for l in trade_belief_state_labels
    }

    # hard-coded list of slot names extracted from `get_slot_information` using the ontology.json file
    all_slots: List[str] = [
        "hotel-pricerange",
        "hotel-type",
        "hotel-parking",
        "hotel-book stay",
        "hotel-book day",
        "hotel-book people",
        "hotel-area",
        "hotel-stars",
        "hotel-internet",
        "hotel-name",
        "train-destination",
        "train-day",
        "train-departure",
        "train-arriveby",
        "train-book people",
        "train-leaveat",
        "restaurant-food",
        "restaurant-pricerange",
        "restaurant-area",
        "restaurant-name",
        "attraction-area",
        "attraction-name",
        "attraction-type",
        "taxi-leaveat",
        "taxi-destination",
        "taxi-departure",
        "taxi-arriveby",
        "restaurant-book time",
        "restaurant-book day",
        "restaurant-book people",
    ]

    general_typo = {
        # type
        "guesthouse": "guest house",
        "guesthouses": "guest house",
        "guest": "guest house",
        "mutiple sports": "multiple sports",
        "sports": "multiple sports",
        "mutliple sports": "multiple sports",
        "swimmingpool": "swimming pool",
        "concerthall": "concert hall",
        "concert": "concert hall",
        "pool": "swimming pool",
        "night club": "nightclub",
        "mus": "museum",
        "ol": "architecture",
        "colleges": "college",
        "coll": "college",
        "architectural": "architecture",
        "musuem": "museum",
        "churches": "church",
        # area
        "center": "centre",
        "center of town": "centre",
        "near city center": "centre",
        "in the north": "north",
        "cen": "centre",
        "east side": "east",
        "east area": "east",
        "west part of town": "west",
        "ce": "centre",
        "town center": "centre",
        "centre of cambridge": "centre",
        "city center": "centre",
        "the south": "south",
        "scentre": "centre",
        "town centre": "centre",
        "in town": "centre",
        "north part of town": "north",
        "centre of town": "centre",
        "cb30aq": "none",
        # price
        "mode": "moderate",
        "moderate -ly": "moderate",
        "mo": "moderate",
        # day
        "next friday": "friday",
        "monda": "monday",
        # parking
        "free parking": "free",
        # internet
        "free internet": "yes",
        # star
        "4 star": "4",
        "4 stars": "4",
        "0 star rarting": "none",
        # others
        "y": "yes",
        "any": "dontcare",
        "n": "no",
        "does not care": "dontcare",
        "not men": "none",
        "not": "none",
        "not mentioned": "none",
        "": "none",
        "not mendtioned": "none",
        "3 .": "3",
        "does not": "no",
        "fun": "none",
        "art": "none",
    }

    # pylint: disable=too-many-boolean-expressions
    for slot in all_slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in general_typo.keys():
                label_dict[slot] = label_dict[slot].replace(
                    label_dict[slot], general_typo[label_dict[slot]]
                )

            # miss match slot and value
            if (
                slot == "hotel-type"
                and label_dict[slot]
                in [
                    "nigh",
                    "moderate -ly priced",
                    "bed and breakfast",
                    "centre",
                    "venetian",
                    "intern",
                    "a cheap -er hotel",
                ]
                or slot == "hotel-internet"
                and label_dict[slot] == "4"
                or slot == "hotel-pricerange"
                and label_dict[slot] == "2"
                or slot == "attraction-type"
                and label_dict[slot]
                in ["gastropub", "la raza", "galleria", "gallery", "science", "m"]
                or "area" in slot
                and label_dict[slot] in ["moderate"]
                or "day" in slot
                and label_dict[slot] == "t"
            ):
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in [
                "hotel with free parking and free wifi",
                "4",
                "3 star hotel",
            ]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no":
                    label_dict[slot] = "north"
                elif label_dict[slot] == "we":
                    label_dict[slot] = "west"
                elif label_dict[slot] == "cent":
                    label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we":
                    label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no":
                    label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if (
                slot == "restaurant-area"
                and label_dict[slot]
                in ["stansted airport", "cambridge", "silver street"]
                or slot == "attraction-area"
                and label_dict[slot]
                in ["norwich", "ely", "museum", "same area as hotel"]
            ):
                label_dict[slot] = "none"

    return label_dict


def normalize_trade_slot_name(name: str) -> str:
    """Normalizes the TRADE slot name to the dataflow slot name.

    Replace whitespace to make it easier to tokenize plans.
    """
    return re.sub(r"(\s)+", "-", name)


def flatten_belief_state(
    belief_state: BeliefState, keep_all_domains: bool, remove_none: bool
) -> Dict[str, str]:
    """Converts the belief state into a flatten dictionary.

    Args:
        belief_state: the TRADE belief state
        keep_all_domains: True if we keep all domains in the belief state; False if we only keep TRADE experiment domains
        remove_none: True if we remove slots with "none" value from the returned belief dict
    Returns:
        the flatten belief state dictionary
    """
    trade_belief_dict: Dict[str, str] = {
        item["slots"][0][0]: item["slots"][0][1] for item in belief_state
    }
    return {
        normalize_trade_slot_name(name=slot_fullname): slot_value
        for slot_fullname, slot_value in trade_belief_dict.items()
        if (not remove_none or slot_value != "none")
        and (
            keep_all_domains
            or get_domain_and_slot_name(slot_fullname)[0] in _EXPERIMENT_DOMAINS
        )
    }
