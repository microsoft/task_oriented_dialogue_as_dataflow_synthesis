#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Creates BeliefStateTrackerDatum from different sources TRADE processed dialogues.
"""

import argparse
import json
from typing import Any, Dict, Iterator, List

from dataflow.core.io_utils import save_jsonl_file
from dataflow.multiwoz.belief_state_tracker_datum import (
    BeliefState,
    BeliefStateTrackerDatum,
    Slot,
    sort_slots,
)
from dataflow.multiwoz.ontology import DATAFLOW_SLOT_NAMES_FOR_DOMAIN
from dataflow.multiwoz.trade_dst_utils import (
    flatten_belief_state,
    get_domain_and_slot_name,
)


def build_belief_state_from_belief_dict(
    belief_dict: Dict[str, str], strict: bool
) -> BeliefState:
    slots_for_domain: Dict[str, List[Slot]] = dict()
    for slot_fullname, slot_value in belief_dict.items():
        domain, slot_name = get_domain_and_slot_name(slot_fullname)
        if strict:
            assert (
                slot_name in DATAFLOW_SLOT_NAMES_FOR_DOMAIN[domain]
            ), 'slot "{}" is not in ontology for domain "{}"'.format(slot_name, domain)
        elif slot_name not in DATAFLOW_SLOT_NAMES_FOR_DOMAIN[domain]:
            # NOTE: We only print a warning. The slot will be still included in the
            # belief state for evaluation.
            # If we assume the Belief State Tracker knows the ontology in advance, then
            # we can remove the slot from the prediction.
            print(
                'slot "{}" is not in ontology for domain "{}"'.format(slot_name, domain)
            )
        if domain not in slots_for_domain:
            slots_for_domain[domain] = []
        slots_for_domain[domain].append(Slot(name=slot_name, value=slot_value))
    sort_slots(slots_for_domain)
    return BeliefState(slots_for_domain=slots_for_domain)


def build_belief_state_from_trade_turn(trade_turn: Dict[str, Any]) -> BeliefState:
    """Returns a BeliefState object from a TRADE turn."""
    # do not drop any slots or change any slot values
    belief_dict = flatten_belief_state(
        belief_state=trade_turn["belief_state"],
        keep_all_domains=True,
        remove_none=False,
    )
    return build_belief_state_from_belief_dict(belief_dict=belief_dict, strict=True)


def build_belief_state_tracker_data_from_trade_dialogue(
    trade_dialogue: Dict[str, Any],
) -> Iterator[BeliefStateTrackerDatum]:
    for trade_turn in trade_dialogue["dialogue"]:
        yield BeliefStateTrackerDatum(
            dialogue_id=trade_dialogue["dialogue_idx"],
            turn_index=int(trade_turn["turn_idx"]),
            belief_state=build_belief_state_from_trade_turn(trade_turn),
            prev_agent_utterance=trade_turn["system_transcript"],
            curr_user_utterance=trade_turn["transcript"],
        )


def main(trade_data_file: str, belief_state_tracker_data_file: str) -> None:
    with open(trade_data_file) as fp:
        trade_dialogues = json.loads(fp.read().strip())
    belief_state_tracker_data = [
        datum
        for trade_dialogue in trade_dialogues
        for datum in build_belief_state_tracker_data_from_trade_dialogue(trade_dialogue)
    ]

    save_jsonl_file(
        data=belief_state_tracker_data,
        data_jsonl=belief_state_tracker_data_file,
        remove_null=True,
    )


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--trade_data_file", help="TRADE processed dialogues file",
    )
    argument_parser.add_argument(
        "--belief_state_tracker_data_file",
        help="output jsonl file of BeliefStateTrackerDatum",
    )


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        trade_data_file=args.trade_data_file,
        belief_state_tracker_data_file=args.belief_state_tracker_data_file,
    )
