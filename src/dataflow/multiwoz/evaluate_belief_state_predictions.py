#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Evaluates belief state tracking predictions.
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, cast

import jsons

from dataflow.core.io_utils import load_jsonl_file_and_build_lookup
from dataflow.multiwoz.create_belief_state_prediction_report import (
    BeliefStatePredictionReportDatum,
)
from dataflow.multiwoz.ontology import DATAFLOW_SLOT_NAMES_FOR_DOMAIN


@dataclass
class EvaluationStats:
    num_total_turns: int = 0
    num_correct_turns: int = 0
    num_correct_turns_after_first_error: int = 0
    num_turns_before_first_error: int = 0
    # key: slot name (either with domain or without domain)
    # value: number of correct turns
    num_correct_turns_for_slot: Dict[str, int] = field(default_factory=dict)
    num_total_dialogues: int = 0
    num_correct_dialogues: int = 0

    @property
    def accuracy(self) -> float:
        if self.num_total_turns == 0:
            return 0
        return self.num_correct_turns / self.num_total_turns

    @property
    def accuracy_for_slot(self) -> Dict[str, float]:
        if self.num_total_turns == 0:
            return {
                slot_name: 0 for slot_name, _ in self.num_correct_turns_for_slot.items()
            }
        return {
            slot_name: num_correct_turns / self.num_total_turns
            for slot_name, num_correct_turns in self.num_correct_turns_for_slot.items()
        }

    @property
    def ave_num_turns_before_first_error(self) -> float:
        if self.num_total_dialogues == 0:
            return 0
        return self.num_turns_before_first_error / self.num_total_dialogues

    @property
    def pct_correct_dialogues(self) -> float:
        if self.num_total_dialogues == 0:
            return 0
        return self.num_correct_dialogues / self.num_total_dialogues

    def __iadd__(self, other: object) -> "EvaluationStats":
        if not isinstance(other, EvaluationStats):
            raise ValueError()
        self.num_total_turns += other.num_total_turns
        self.num_correct_turns += other.num_correct_turns
        self.num_correct_turns_after_first_error += (
            other.num_correct_turns_after_first_error
        )
        self.num_turns_before_first_error += other.num_turns_before_first_error
        self.num_correct_turns_for_slot = cast(
            Dict[str, int], self.num_correct_turns_for_slot
        )
        other.num_correct_turns_for_slot = cast(
            Dict[str, int], other.num_correct_turns_for_slot
        )
        for (slot_name, num_correct_turns,) in other.num_correct_turns_for_slot.items():
            if slot_name not in self.num_correct_turns_for_slot:
                self.num_correct_turns_for_slot[slot_name] = 0
            self.num_correct_turns_for_slot[slot_name] += num_correct_turns
        self.num_total_dialogues += other.num_total_dialogues
        self.num_correct_dialogues += other.num_correct_dialogues

        return self

    def __add__(self, other: object) -> "EvaluationStats":
        if not isinstance(other, EvaluationStats):
            raise ValueError()
        result = EvaluationStats()
        result += self
        result += other

        return result


def evaluate_dialogue(
    dialogue: Dict[int, BeliefStatePredictionReportDatum],
) -> EvaluationStats:
    """Evaluates a dialogue."""
    stats = EvaluationStats()
    seen_error = False
    prediction_report_datum: BeliefStatePredictionReportDatum
    for turn_index, prediction_report_datum in sorted(
        dialogue.items(), key=lambda x: int(x[0])
    ):
        assert turn_index == prediction_report_datum.turn_index

        stats.num_total_turns += 1

        if prediction_report_datum.is_correct:
            stats.num_correct_turns += 1
            if seen_error:
                stats.num_correct_turns_after_first_error += 1
            else:
                stats.num_turns_before_first_error += 1
        else:
            seen_error = True

        for domain, slot_names in DATAFLOW_SLOT_NAMES_FOR_DOMAIN.items():
            gold_slots = prediction_report_datum.gold.slots_for_domain.get(domain, [])
            gold_slot_value_lookup = {slot.name: slot.value for slot in gold_slots}
            assert len(gold_slot_value_lookup) == len(gold_slots)

            hypo_slots = prediction_report_datum.prediction.slots_for_domain.get(
                domain, []
            )
            hypo_slot_value_lookup = {slot.name: slot.value for slot in hypo_slots}
            assert len(hypo_slot_value_lookup) == len(hypo_slots)

            for slot_name in slot_names:
                gold_slot_value = gold_slot_value_lookup.get(slot_name)
                hypo_slot_value = hypo_slot_value_lookup.get(slot_name)

                # these two values should be treated as null and the slots should not be presented in the belief state
                assert gold_slot_value not in ["", "not mentioned"]
                assert hypo_slot_value not in ["", "not mentioned"]

                if gold_slot_value != hypo_slot_value:
                    continue

                slot_fullname = "{}-{}".format(domain, slot_name)
                if slot_fullname not in stats.num_correct_turns_for_slot:
                    stats.num_correct_turns_for_slot[slot_fullname] = 0
                stats.num_correct_turns_for_slot[slot_fullname] += 1

    if not seen_error:
        stats.num_correct_dialogues += 1
    stats.num_total_dialogues += 1

    return stats


def evaluate_dataset(
    prediction_report_lookup: Dict[str, Dict[int, BeliefStatePredictionReportDatum]],
) -> EvaluationStats:
    evaluation_stats = EvaluationStats()
    for _dialogue_id, dialogue in prediction_report_lookup.items():
        stats = evaluate_dialogue(dialogue=dialogue)
        evaluation_stats += stats

    return evaluation_stats


def main(prediction_report_jsonl: str, outbase: str) -> str:
    prediction_report_lookup = load_jsonl_file_and_build_lookup(
        data_jsonl=prediction_report_jsonl,
        cls=BeliefStatePredictionReportDatum,
        primary_key_getter=lambda x: x.dialogue_id,
        secondary_key_getter=lambda x: x.turn_index,
    )

    stats = evaluate_dataset(prediction_report_lookup=prediction_report_lookup)
    scores_file = outbase + ".scores.json"
    with open(scores_file, "w") as fp:
        fp.write(jsons.dumps(stats, {"indent": 2, "sort_keys": True}))
        fp.write("\n")

    return scores_file


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--prediction_report_jsonl", help="the prediction report jsonl file"
    )
    argument_parser.add_argument("--outbase", help="output files basename")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(prediction_report_jsonl=args.prediction_report_jsonl, outbase=args.outbase)
