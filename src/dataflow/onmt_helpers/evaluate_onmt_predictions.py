#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Evaluates 1best predictions.

Computes both turn-level and dialogue-level accuracy.
"""

import argparse
import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jsons
import pandas as pd

from dataflow.core.dialogue import TurnId
from dataflow.core.io import load_jsonl_file


@dataclass
class EvaluationScores:
    num_total_turns: int = 0
    num_correct_turns: int = 0
    num_turns_before_first_error: int = 0
    num_total_dialogues: int = 0
    num_correct_dialogues: int = 0

    @property
    def accuracy(self) -> float:
        if self.num_total_turns == 0:
            return 0
        return self.num_correct_turns / self.num_total_turns

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

    def __iadd__(self, other: object) -> "EvaluationScores":
        if not isinstance(other, EvaluationScores):
            raise ValueError()
        self.num_total_turns += other.num_total_turns
        self.num_correct_turns += other.num_correct_turns
        self.num_turns_before_first_error += other.num_turns_before_first_error
        self.num_total_dialogues += other.num_total_dialogues
        self.num_correct_dialogues += other.num_correct_dialogues

        return self

    def __add__(self, other: object) -> "EvaluationScores":
        if not isinstance(other, EvaluationScores):
            raise ValueError()
        result = EvaluationScores()
        result += self
        result += other

        return result


def evaluate_dialogue(turns: List[Tuple[int, bool]]) -> EvaluationScores:
    num_correct_turns = 0
    dialogue_is_correct = True
    num_turns_before_first_error = 0
    seen_error = False
    for _turn_index, is_correct in sorted(turns, key=lambda x: x[0]):
        if is_correct:
            num_correct_turns += 1
            if not seen_error:
                num_turns_before_first_error += 1
        else:
            dialogue_is_correct = False
            seen_error = True

    return EvaluationScores(
        num_total_turns=len(turns),
        num_correct_turns=num_correct_turns,
        num_turns_before_first_error=num_turns_before_first_error,
        num_total_dialogues=1,
        num_correct_dialogues=1 if dialogue_is_correct else 0,
    )


def evaluate_dataset(
    prediction_report_df: pd.DataFrame, use_leaderboard_metric: bool
) -> EvaluationScores:
    # pylint: disable=singleton-comparison
    dataset_scores = EvaluationScores()
    if use_leaderboard_metric:
        field_name = "isCorrectLeaderboard"
    else:
        field_name = "isCorrect"
    for _dialogue_id, df_for_dialogue in prediction_report_df.groupby("dialogueId"):
        turns = [
            (int(row.get("turnIndex")), row.get(field_name))
            for _, row in df_for_dialogue.iterrows()
        ]
        dialogue_scores = evaluate_dialogue(turns)
        dataset_scores += dialogue_scores

    return dataset_scores


def main(
    prediction_report_tsv: str,
    datum_ids_jsonl: Optional[str],
    use_leaderboard_metric: bool,
    scores_json: str,
) -> None:
    prediction_report_df = pd.read_csv(
        prediction_report_tsv,
        sep="\t",
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        na_values=None,
        keep_default_na=False,
    )
    assert not prediction_report_df.isnull().any().any()

    if datum_ids_jsonl:
        datum_ids = set(
            load_jsonl_file(data_jsonl=datum_ids_jsonl, cls=TurnId, verbose=False)
        )
        mask_datum_id = [
            TurnId(dialogue_id=row.get("dialogueId"), turn_index=row.get("turnIndex"))
            in datum_ids
            for _, row in prediction_report_df.iterrows()
        ]
        prediction_report_df = prediction_report_df.loc[mask_datum_id]

    scores = evaluate_dataset(prediction_report_df, use_leaderboard_metric)
    with open(scores_json, "w") as fp:
        fp.write(jsons.dumps(scores, jdkwargs={"indent": 2}))
        fp.write("\n")


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--prediction_report_tsv", help="the prediction report tsv file"
    )
    argument_parser.add_argument(
        "--datum_ids_jsonl", default=None, help="if set, only evaluate on these turns",
    )
    argument_parser.add_argument(
        "--use_leaderboard_metric",
        default=False,
        action="store_true",
        help="if set, use the isCorrectLeaderboard field instead of isCorrect field in the prediction report",
    )
    argument_parser.add_argument("--scores_json", help="output scores json file")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    if not args.use_leaderboard_metric:
        print(
            "WARNING: The flag --use_leaderboard_metric is not set."
            " The reported results will be consistent with the numbers"
            " reported in the TACL2020 paper. To report on the leaderboard evaluation metric, please use"
            " --use_leaderboard_metric, which canonicalizes the labels and predictions."
        )
    main(
        prediction_report_tsv=args.prediction_report_tsv,
        datum_ids_jsonl=args.datum_ids_jsonl,
        use_leaderboard_metric=args.use_leaderboard_metric,
        scores_json=args.scores_json,
    )
