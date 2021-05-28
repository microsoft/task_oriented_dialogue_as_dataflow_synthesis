#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Evaluation script for the leaderboard.
"""
import argparse
import json
from typing import Iterable, List, Optional, Set, Tuple

from dataflow.core.dialogue import TurnId
from dataflow.core.io import load_jsonl_file
from dataflow.core.lispress import try_round_trip
from dataflow.core.turn_prediction import TurnAnswer, TurnPrediction, missing_prediction


def evaluate_prediction_exact_match(
    pred: TurnPrediction, gold: TurnAnswer
) -> Tuple[bool, bool]:
    assert pred.datum_id == gold.datum_id, f"mismatched data: {pred}, {gold}"
    pred_lispress = try_round_trip(pred.lispress)
    gold_lispress = try_round_trip(gold.lispress)
    if pred_lispress != gold_lispress:
        print(
            f"Misprediction on {gold.datum_id.dialogue_id}:{gold.datum_id.turn_index} | {gold.user_utterance}\nPred: {pred_lispress}\nGold: {gold_lispress}\n"
        )
    elif not gold.program_execution_oracle.refer_are_correct:
        print(
            f"Example {gold.datum_id.dialogue_id}:{gold.datum_id.turn_index} can't be correct because the refer call is not correct.\n"
        )
    return (
        pred_lispress == gold_lispress
        and gold.program_execution_oracle.refer_are_correct,
        pred_lispress == gold_lispress,
    )


def evaluate_predictions_exact_match(
    preds_and_golds: Iterable[Tuple[TurnPrediction, TurnAnswer]]
) -> Tuple[float, float]:
    correct = 0
    correct_ignoring_refer = 0
    total = 0
    for pred, gold in preds_and_golds:
        total += 1
        (right, right_ignoring_refer) = evaluate_prediction_exact_match(pred, gold)
        correct += int(right)
        correct_ignoring_refer += int(right_ignoring_refer)

    return (
        correct / total if total else 0,
        correct_ignoring_refer / total if total else 0,
    )


def collate(
    preds: List[TurnPrediction],
    golds: List[TurnAnswer],
    datum_ids: Optional[Set[TurnId]],
) -> List[Tuple[TurnPrediction, TurnAnswer]]:
    """
    For each datum `gold` in `golds`, if `gold.datum_id` is in `datum_ids`,
    return a tuple of `(pred, gold)`, where `pred` is in `preds` and
    `pred.datum_id == gold.datum_id`
    If no such `pred` exists, `gold` is paired with a special "missing"
    prediction which is never correct.
    """
    pred_by_id = {pred.datum_id: pred for pred in preds}
    pred_ids = set(pred_by_id.keys())
    gold_ids = {gold.datum_id for gold in golds}
    if datum_ids is not None:
        gold_ids &= datum_ids
    missing_ids = gold_ids - pred_ids
    extra_ids = pred_ids - gold_ids
    if missing_ids:
        print(f"Gold turns not predicted: {list(missing_ids)}")
    if extra_ids:
        pass
    return [
        (pred_by_id.get(gold.datum_id, missing_prediction(gold.datum_id)), gold)
        for gold in golds
        if datum_ids is None or gold.datum_id in datum_ids
    ]


def evaluate_prediction_file(
    predictions_jsonl: str, gold_jsonl: str, datum_ids_jsonl: Optional[str]
) -> Tuple[float, float]:
    preds = list(load_jsonl_file(predictions_jsonl, TurnPrediction, verbose=False))
    golds = list(load_jsonl_file(gold_jsonl, TurnAnswer, verbose=False))
    datum_ids = (
        None
        if datum_ids_jsonl is None
        else set(load_jsonl_file(data_jsonl=datum_ids_jsonl, cls=TurnId, verbose=False))
    )
    return evaluate_predictions_exact_match(collate(preds, golds, datum_ids))


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--predictions_jsonl", help="the predictions jsonl file to evaluate",
    )
    argument_parser.add_argument(
        "--gold_jsonl", help="the gold jsonl file to evaluate against",
    )
    argument_parser.add_argument(
        "--datum_ids_jsonl", default=None, help="if set, only evaluate on these turns",
    )
    argument_parser.add_argument("--scores_json", help="output scores json file")


def write_accuracy_json(
    accuracies: Tuple[float, float], scores_json_filename: str
) -> None:
    (accuracy, accuracy_ignoring_refer) = accuracies
    with open(scores_json_filename, mode="w", encoding="utf8") as scores_json_file:
        scores_json_file.write(
            json.dumps(
                {
                    "accuracy": accuracy,
                    "accuracy_ignorning_refer": accuracy_ignoring_refer,
                }
            )
        )


def main():
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    accuracies = evaluate_prediction_file(
        predictions_jsonl=args.predictions_jsonl,
        gold_jsonl=args.gold_jsonl,
        datum_ids_jsonl=args.datum_ids_jsonl,
    )
    write_accuracy_json(accuracies, args.scores_json)


if __name__ == "__main__":
    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main()
