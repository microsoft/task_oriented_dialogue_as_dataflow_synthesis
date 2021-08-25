#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Creates the belief state prediction report (List[BeliefStatePredictionReportDatum]) from either
TRADE predictions (trade) or dataflow execution results (dataflow).
"""
import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from dataflow.core.io_utils import load_jsonl_file_and_build_lookup, save_jsonl_file
from dataflow.core.prediction_report import (
    PredictionReportDatum,
    save_prediction_report_tsv,
    save_prediction_report_txt,
)
from dataflow.multiwoz.belief_state_tracker_datum import (
    BeliefState,
    BeliefStateTrackerDatum,
    Slot,
    pretty_print_belief_state,
    sort_slots,
)
from dataflow.multiwoz.create_belief_state_tracker_data import (
    build_belief_state_from_belief_dict,
)
from dataflow.multiwoz.execute_programs import CompleteExecutionResult
from dataflow.multiwoz.ontology import DATAFLOW_SLOT_NAMES_FOR_DOMAIN
from dataflow.multiwoz.trade_dst_utils import normalize_trade_slot_name


@dataclass(frozen=True)
class BeliefStatePredictionReportDatum(PredictionReportDatum):
    dialogue_id: str
    turn_index: int
    prev_agent_utterance: str
    curr_user_utterance: str
    gold: BeliefState
    prediction: BeliefState

    @property
    def is_correct(self) -> bool:
        return self.gold == self.prediction

    def flatten(self) -> Dict[str, Union[str, int]]:
        return {
            "dialogueId": self.dialogue_id,
            "turnIndex": self.turn_index,
            "isCorrect": self.is_correct,
            "prevAgentUtterance": self.prev_agent_utterance,
            "currUserUtterance": self.curr_user_utterance,
            "gold": str(self.gold),
            "prediction": str(self.prediction),
        }


def get_belief_state_without_none(belief_state: BeliefState) -> BeliefState:
    filtered_belief_state = BeliefState(slots_for_domain={})
    for domain, slots in belief_state.slots_for_domain.items():
        filtered_slots = [slot for slot in slots if slot.value != "none"]
        if filtered_slots:
            filtered_belief_state.slots_for_domain[domain] = filtered_slots
    return filtered_belief_state


def build_belief_state_from_trade_prediction(
    trade_prediction: Dict[str, Any], use_gold: bool
) -> BeliefState:
    """Returns a BeliefState object from a TRADE prediction."""
    if use_gold:
        hypo_belief_state = trade_prediction["turn_belief"]
    else:
        hypo_belief_state = trade_prediction["pred_bs_ptr"]
    slots_for_domain: Dict[str, List[Slot]] = dict()
    for item in hypo_belief_state:
        domain, slot_name, slot_value = item.split("-")

        slot_name = normalize_trade_slot_name(slot_name)
        assert (
            slot_name in DATAFLOW_SLOT_NAMES_FOR_DOMAIN[domain]
        ), 'slot "{}" is not in ontology for domain "{}"'.format(slot_name, domain)
        if domain not in slots_for_domain:
            slots_for_domain[domain] = []
        slots_for_domain[domain].append(Slot(name=slot_name, value=slot_value))
    sort_slots(slots_for_domain)
    return BeliefState(slots_for_domain=slots_for_domain)


def build_prediction_report_datum_from_trade_prediction(
    dialogue_id: str,
    turn_idx: int,
    gold_belief_state_tracker_data: Dict[str, Dict[int, BeliefStateTrackerDatum]],
    trade_prediction: Dict[str, Any],
    remove_none: bool,
) -> BeliefStatePredictionReportDatum:
    gold_belief_state_tracker_datum = gold_belief_state_tracker_data[dialogue_id][
        turn_idx
    ]
    gold_belief_state_in_trade_prediction = build_belief_state_from_trade_prediction(
        trade_prediction, True
    )
    if (
        gold_belief_state_in_trade_prediction
        != gold_belief_state_tracker_datum.belief_state
    ):
        print(f"gold belief state mismatch {dialogue_id} {turn_idx}")
        print("====Gold in TRADE Prediction====")
        print(pretty_print_belief_state(gold_belief_state_in_trade_prediction))
        print("====Gold in BeliefStateTrackerDatum====")
        print(pretty_print_belief_state(gold_belief_state_tracker_datum.belief_state))
        raise ValueError(f"gold belief state mismatch {dialogue_id} {turn_idx}")

    hypo_belief_state = build_belief_state_from_trade_prediction(
        trade_prediction, False
    )
    if remove_none:
        # NOTE: Only removes "none" in predicted belief state; the gold should remain unchanged.
        hypo_belief_state = get_belief_state_without_none(hypo_belief_state)

    return BeliefStatePredictionReportDatum(
        dialogue_id=dialogue_id,
        turn_index=int(turn_idx),
        prev_agent_utterance=gold_belief_state_tracker_datum.prev_agent_utterance,
        curr_user_utterance=gold_belief_state_tracker_datum.curr_user_utterance,
        gold=gold_belief_state_tracker_datum.belief_state,
        prediction=hypo_belief_state,
    )


def build_prediction_report_from_trade_predictions(
    trade_predictions_file: str,
    gold_belief_state_tracker_data: Dict[str, Dict[int, BeliefStateTrackerDatum]],
    remove_none: bool,
) -> List[BeliefStatePredictionReportDatum]:
    with open(trade_predictions_file) as fp:
        trade_predictions = json.loads(fp.read())
    return [
        build_prediction_report_datum_from_trade_prediction(
            dialogue_id=dialogue_id,
            turn_idx=int(turn_idx),
            gold_belief_state_tracker_data=gold_belief_state_tracker_data,
            trade_prediction=prediction,
            remove_none=remove_none,
        )
        for dialogue_id, predictions in trade_predictions.items()
        for turn_idx, prediction in predictions.items()
    ]


def build_prediction_report_datum_from_execution_results(
    dialogue_id: str,
    turn_idx: int,
    gold_belief_state_tracker_data: Dict[str, Dict[int, BeliefStateTrackerDatum]],
    execution_result_for_turn: CompleteExecutionResult,
    remove_none: bool,
) -> BeliefStatePredictionReportDatum:
    gold_belief_state_tracker_datum = gold_belief_state_tracker_data[dialogue_id][
        turn_idx
    ]
    hypo_belief_state = build_belief_state_from_belief_dict(
        belief_dict=execution_result_for_turn.belief_dict, strict=False
    )
    if remove_none:
        # only removes "none" in predicted belief state; the gold_belief_state should NOT be changed
        hypo_belief_state = get_belief_state_without_none(hypo_belief_state)
    return BeliefStatePredictionReportDatum(
        dialogue_id=dialogue_id,
        turn_index=int(turn_idx),
        prev_agent_utterance=gold_belief_state_tracker_datum.prev_agent_utterance,
        curr_user_utterance=gold_belief_state_tracker_datum.curr_user_utterance,
        gold=gold_belief_state_tracker_datum.belief_state,
        prediction=hypo_belief_state,
    )


def build_prediction_report_from_dataflow_execution_results(
    dataflow_execution_results_file: str,
    gold_belief_state_tracker_data: Dict[str, Dict[int, BeliefStateTrackerDatum]],
    remove_none: bool,
) -> List[BeliefStatePredictionReportDatum]:
    execution_results = load_jsonl_file_and_build_lookup(
        data_jsonl=dataflow_execution_results_file,
        cls=CompleteExecutionResult,
        primary_key_getter=lambda x: x.dialogue_id,
        secondary_key_getter=lambda x: x.turn_index,
        unit=" turns",
    )
    return [
        build_prediction_report_datum_from_execution_results(
            dialogue_id=dialogue_id,
            turn_idx=turn_idx,
            gold_belief_state_tracker_data=gold_belief_state_tracker_data,
            execution_result_for_turn=execution_result_for_turn,
            remove_none=remove_none,
        )
        for dialogue_id, execution_results_for_dialogue in execution_results.items()
        for (
            turn_idx,
            execution_result_for_turn,
        ) in execution_results_for_dialogue.items()
    ]


def main(
    input_data_file: str,
    file_format: str,
    remove_none: bool,
    gold_data_file: str,
    outbase: str,
) -> str:
    gold_belief_state_tracker_data: Dict[
        str, Dict[int, BeliefStateTrackerDatum]
    ] = load_jsonl_file_and_build_lookup(
        data_jsonl=gold_data_file,
        cls=BeliefStateTrackerDatum,
        primary_key_getter=lambda x: x.dialogue_id,
        secondary_key_getter=lambda x: x.turn_index,
        unit=" turns",
    )

    if file_format == "trade":
        prediction_report = build_prediction_report_from_trade_predictions(
            trade_predictions_file=input_data_file,
            gold_belief_state_tracker_data=gold_belief_state_tracker_data,
            remove_none=remove_none,
        )
    elif file_format == "dataflow":
        prediction_report = build_prediction_report_from_dataflow_execution_results(
            dataflow_execution_results_file=input_data_file,
            gold_belief_state_tracker_data=gold_belief_state_tracker_data,
            remove_none=remove_none,
        )
    else:
        raise ValueError(f"unsupported file format: {file_format}")

    belief_state_prediction_report_jsonl = f"{outbase}.prediction_report.jsonl"
    save_jsonl_file(
        data=prediction_report,
        data_jsonl=belief_state_prediction_report_jsonl,
        remove_null=True,
    )

    belief_state_prediction_report_tsv = f"{outbase}.prediction_report.tsv"
    save_prediction_report_tsv(prediction_report, belief_state_prediction_report_tsv)

    belief_state_prediction_report_txt = f"{outbase}.prediction_report.txt"
    save_prediction_report_txt(
        prediction_report,
        belief_state_prediction_report_txt,
        [
            "dialogueId",
            "turnIndex",
            "isCorrect",
            "prevAgentUtterance",
            "currUserUtterance",
            "gold",
            "prediction",
        ],
    )

    return belief_state_prediction_report_jsonl


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--input_data_file", help="input data file",
    )
    argument_parser.add_argument(
        "--format",
        default=None,
        choices=["trade", "dataflow"],
        help="input data file format",
    )
    argument_parser.add_argument(
        "--gold_data_file",
        default=None,
        help="gold belief state tracker data file (optional)",
    )
    argument_parser.add_argument(
        "--remove_none",
        default=False,
        action="store_true",
        help='if True, remove "none" slot values in the belief state',
    )
    argument_parser.add_argument("--outbase", help="output files basename")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        input_data_file=args.input_data_file,
        file_format=args.format,
        remove_none=args.remove_none,
        gold_data_file=args.gold_data_file,
        outbase=args.outbase,
    )
