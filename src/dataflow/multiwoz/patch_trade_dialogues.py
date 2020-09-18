#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Patches TRADE-processed dialogues.

In TRADE, there are extra steps to fix belief state labels after the data are dumped from `create_data.py`.
It makes the evaluation and comparison difficult b/c those label correction and evaluation are embedded in the training
code rather than separate CLI scripts.
This new script applies the TRADE label corrections (fix_general_label_errors) and re-dumps the dialogues in the same format:

NOTE: This only patches the "belief_state". Other fields including "turn_label" are unchanged. Thus, there can be
inconsistency between "belief_state" and "turn_label".
"""
import argparse
import json
from typing import Dict, List, Tuple

from dataflow.multiwoz.ontology import TRADE_SLOT_NAMES_FOR_DOMAIN
from dataflow.multiwoz.trade_dst_utils import (
    fix_general_label_error,
    get_domain_and_slot_name,
)


def validate_belief_dict(trade_belief_dict: Dict[str, str]) -> List[str]:
    """Validates the belief dict and returns the domains with all "none" values."""
    slots_lookup: Dict[str, Dict[str, str]] = {}
    for slot_fullname, slot_value in trade_belief_dict.items():
        # slot value should not be "empty" or "not mentioned"
        assert slot_value not in ["", "not mentioned"]

        domain, slot_name = get_domain_and_slot_name(slot_fullname)
        assert slot_name in TRADE_SLOT_NAMES_FOR_DOMAIN[domain]

        if domain not in slots_lookup:
            slots_lookup[domain] = {}
        slots_lookup[domain][slot_name] = slot_value

    # active domains should have at least one slot that is not "none"
    all_none_domains = []
    for domain, slots in slots_lookup.items():
        if all([slot_value == "none" for slot_value in slots.values()]):
            all_none_domains.append(domain)
    return all_none_domains


def main(trade_data_file: str, outbase: str) -> Tuple[str, str]:
    trade_dialogues = json.load(open(trade_data_file, "r"))
    # turns that need manual review
    need_review_turns = []
    for trade_dialogue in trade_dialogues:
        for trade_turn in trade_dialogue["dialogue"]:
            trade_belief_dict = fix_general_label_error(trade_turn["belief_state"])
            for item in trade_turn["belief_state"]:
                assert item["act"] == "inform"
            all_none_domains = validate_belief_dict(trade_belief_dict)
            if all_none_domains:
                is_last_turn = int(trade_turn["turn_idx"]) + 1 == len(
                    trade_dialogue["dialogue"]
                )
                need_review_turns.append(
                    {
                        "dialogueId": trade_dialogue["dialogue_idx"],
                        "turnIndex": trade_turn["turn_idx"],
                        "isLastTurn": is_last_turn,
                        "prevAgentUtterance": trade_turn["system_transcript"],
                        "currUserUtterance": trade_turn["transcript"],
                        "beliefDict": trade_belief_dict,
                        "allNoneDomains": all_none_domains,
                    }
                )

            trade_turn["belief_state"] = [
                {"slots": [[slot_fullname, slot_value]], "act": "inform"}
                for slot_fullname, slot_value in trade_belief_dict.items()
            ]

    patched_dials_file = outbase + "_dials.json"
    with open(patched_dials_file, "w") as fp:
        json.dump(trade_dialogues, fp, indent=4)

    need_review_turns_file = outbase + "_need_review_turns.json"
    with open(need_review_turns_file, "w") as fp:
        json.dump(need_review_turns, fp, indent=2)

    return patched_dials_file, need_review_turns_file


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    cmdline_parser.add_argument("--trade_data_file", help="the trade data file")
    cmdline_parser.add_argument("--outbase", help="output files base name")
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(trade_data_file=args.trade_data_file, outbase=args.outbase)
