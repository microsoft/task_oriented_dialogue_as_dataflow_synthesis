#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from typing import Dict, List

from dataflow.multiwoz.ontology import DATAFLOW_SLOT_NAMES_FOR_DOMAIN
from dataflow.multiwoz.trade_dst_utils import (
    get_domain_and_slot_name,
    normalize_trade_slot_name,
    trade_normalize_slot_name,
)


def expected_dataflow_slot_names_for_domain() -> Dict[str, List[str]]:
    # extracted from MultiWoZ-2.1 ontology.json file
    # $ jq 'keys' ontology.json
    raw_slot_fullnames: List[str] = [
        "attraction-area",
        "attraction-name",
        "attraction-type",
        "hotel-area",
        "hotel-book day",
        "hotel-book people",
        "hotel-book stay",
        "hotel-internet",
        "hotel-name",
        "hotel-parking",
        "hotel-price range",
        "hotel-stars",
        "hotel-type",
        "restaurant-area",
        "restaurant-book day",
        "restaurant-book people",
        "restaurant-book time",
        "restaurant-food",
        "restaurant-name",
        "restaurant-price range",
        "taxi-arrive by",
        "taxi-departure",
        "taxi-destination",
        "taxi-leave at",
        "train-arrive by",
        "train-book people",
        "train-day",
        "train-departure",
        "train-destination",
        "train-leave at",
        "bus-day",
        "bus-departure",
        "bus-destination",
        "bus-leaveAt",
        "hospital-department",
    ]

    dataflow_slot_fullnames_for_domain: Dict[str, List[str]] = dict()
    for slot_fullname in sorted(raw_slot_fullnames):
        slot_fullname = normalize_trade_slot_name(
            name=trade_normalize_slot_name(name=slot_fullname)
        )
        domain, slot_name = get_domain_and_slot_name(slot_fullname=slot_fullname)
        if domain not in dataflow_slot_fullnames_for_domain:
            dataflow_slot_fullnames_for_domain[domain] = []
        dataflow_slot_fullnames_for_domain[domain].append(slot_name)
    return dataflow_slot_fullnames_for_domain


def test_dataflow_slot_fullnames_for_domain():
    assert DATAFLOW_SLOT_NAMES_FOR_DOMAIN == expected_dataflow_slot_names_for_domain()
