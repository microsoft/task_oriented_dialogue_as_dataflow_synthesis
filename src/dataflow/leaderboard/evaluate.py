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
from dataflow.core.lispress import (
    lispress_to_program,
    parse_lispress,
    program_to_lispress,
    render_compact,
)
from dataflow.core.turn_prediction import TurnAnswer, TurnPrediction, missing_prediction


def _try_round_trip(lispress_str: str) -> str:
    """
    If `lispress_str` is valid lispress, round-trips it to and from `Program`.
    This puts named arguments in alphabetical order.
    If it is not valid, returns the original string unmodified.
    """
    try:
        # round-trip to canonicalize
        lispress = parse_lispress(lispress_str)
        program, _ = lispress_to_program(lispress, 0)
        round_tripped = program_to_lispress(program)
        return render_compact(round_tripped)
    except Exception:  # pylint: disable=W0703
        return lispress_str


def evaluate_prediction_exact_match(pred: TurnPrediction, gold: TurnAnswer) -> bool:
    assert pred.datum_id == gold.datum_id, f"mismatched data: {pred}, {gold}"
    pred_lispress = _try_round_trip(pred.lispress)
    gold_lispress = _try_round_trip(gold.lispress)
    return (
        pred_lispress == gold_lispress
        and gold.program_execution_oracle.refer_are_correct
    )


def evaluate_predictions_exact_match(
    preds_and_golds: Iterable[Tuple[TurnPrediction, TurnAnswer]]
) -> float:
    correct = 0
    total = 0
    for pred, gold in preds_and_golds:
        total += 1
        correct += int(evaluate_prediction_exact_match(pred, gold))

    return correct / total if total else 0


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
) -> float:
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


def write_accuracy_json(accuracy: float, scores_json_filename: str) -> None:
    with open(scores_json_filename, mode="w", encoding="utf8") as scores_json_file:
        scores_json_file.write(json.dumps({"accuracy": accuracy}))


def main():
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    accuracy = evaluate_prediction_file(
        predictions_jsonl=args.predictions_jsonl,
        gold_jsonl=args.gold_jsonl,
        datum_ids_jsonl=args.datum_ids_jsonl,
    )
    write_accuracy_json(accuracy, args.scores_json)


if __name__ == "__main__":
    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main()
