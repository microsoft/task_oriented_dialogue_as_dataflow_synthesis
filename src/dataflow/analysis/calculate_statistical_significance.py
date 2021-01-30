#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Calculates statistical significance for predictions from two experiments.
"""
import argparse
import csv
import json
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from dataflow.core.dialogue import TurnId
from dataflow.core.io import load_jsonl_file
from dataflow.onmt_helpers.evaluate_onmt_predictions import evaluate_dialogue


def get_report_dataframes(
    exp0_prediction_report_df: pd.DataFrame, exp1_prediction_report_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns the turn-level and dialogue-level report dataframes."""
    exp0_prediction_report_df.set_index(
        ["dialogueId", "turnIndex"], inplace=True, drop=True
    )
    exp1_prediction_report_df.set_index(
        ["dialogueId", "turnIndex"], inplace=True, drop=True
    )
    turn_report_df = exp0_prediction_report_df.join(
        exp1_prediction_report_df.loc[:, ["isCorrect"]],
        how="outer",
        lsuffix="_0",
        rsuffix="_1",
    )
    assert not turn_report_df.isnull().any().any()
    assert (
        len(turn_report_df)
        == len(exp0_prediction_report_df)
        == len(exp1_prediction_report_df)
    )

    rows = []
    for dialogue_id, df_for_dialogue in turn_report_df.groupby("dialogueId"):
        dialogue_scores0 = evaluate_dialogue(
            turns=[
                (turn_index, row.get("isCorrect_0"))
                for (_, turn_index), row in df_for_dialogue.iterrows()
            ]
        )
        dialogue_scores1 = evaluate_dialogue(
            turns=[
                (turn_index, row.get("isCorrect_1"))
                for (_, turn_index), row in df_for_dialogue.iterrows()
            ]
        )

        rows.append(
            {
                "dialogueId": dialogue_id,
                "isCorrect_0": dialogue_scores0.num_correct_dialogues > 0,
                "isCorrect_1": dialogue_scores1.num_correct_dialogues > 0,
                "prefix_0": dialogue_scores0.num_turns_before_first_error,
                "prefix_1": dialogue_scores1.num_turns_before_first_error,
            }
        )
    dialogue_report_df = pd.DataFrame(rows)
    return turn_report_df, dialogue_report_df


def run_mcnemar_test(report_df: pd.DataFrame) -> Tuple[float, float]:
    mask_correct_0 = report_df.loc[:, "isCorrect_0"]
    mask_correct_1 = report_df.loc[:, "isCorrect_1"]
    contingency_table = (
        (
            (mask_correct_0 & mask_correct_1).sum(),
            (mask_correct_0 & ~mask_correct_1).sum(),
        ),
        (
            (~mask_correct_0 & mask_correct_1).sum(),
            (~mask_correct_0 & ~mask_correct_1).sum(),
        ),
    )
    result = mcnemar(contingency_table)

    return result.statistic, result.pvalue


def run_paired_permutation_test(
    xs: List[int],
    ys: List[int],
    samples: int = 10000,
    statistic: Callable[[List[int]], float] = np.mean,  # type: ignore
) -> float:
    """Runs the two-sample permutation test to check whether the paired data xs and ys are from the same distribution (null hypothesis).

    Args:
        xs: the data from distribution F1
        ys: the data from distribution F2
        samples: the number of samples for the Monte Carlo sampling
        statistic: the statistic to be used for the test (default is the mean)

    Returns:
        the p-value of the null hypothesis (two-tailed)
    """

    def effect(xx: List[int], yy: List[int]) -> float:
        return np.abs(statistic(xx) - statistic(yy))

    n, k = len(xs), 0
    diff = effect(xs, ys)  # observed difference
    for _ in range(samples):  # for each random sample
        swaps = np.random.randint(0, 2, n).astype(bool)  # flip n coins
        k += diff <= effect(
            np.select([swaps, ~swaps], [xs, ys]),  # swap elements accordingly
            np.select([~swaps, swaps], [xs, ys]),
        )

    # fraction of random samples that achieved at least the observed difference
    return k / float(samples)


def main(
    exp0_prediction_report_tsv: str,
    exp1_prediction_report_tsv: str,
    datum_ids_jsonl: Optional[str],
    scores_json: str,
) -> None:
    """Loads the two prediction report files and calculates statistical significance.

    For the turn-level and dialogue-level accuracy, we use the McNemar test.
    For the dialogue-level prefix length (i.e., the number of turns before the first error), we use the two-sample permutation test.

    If `datum_ids_jsonl` is given, we only use the subset of turns specified in the file. In this case, only turn-level
    metrics are used since it doesn't make sense to compute dialogue-level metrics with only a subset of turns.
    """
    exp0_prediction_report_df = pd.read_csv(
        exp0_prediction_report_tsv,
        sep="\t",
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        na_values=None,
        keep_default_na=False,
    )
    assert not exp0_prediction_report_df.isnull().any().any()

    exp1_prediction_report_df = pd.read_csv(
        exp1_prediction_report_tsv,
        sep="\t",
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        na_values=None,
        keep_default_na=False,
    )
    assert not exp1_prediction_report_df.isnull().any().any()

    turn_report_df, dialogue_report_df = get_report_dataframes(
        exp0_prediction_report_df=exp0_prediction_report_df,
        exp1_prediction_report_df=exp1_prediction_report_df,
    )

    if not datum_ids_jsonl:
        turn_statistic, turn_pvalue = run_mcnemar_test(turn_report_df)
        dialogue_statistic, dialogue_pvalue = run_mcnemar_test(dialogue_report_df)
        prefix_pvalue = run_paired_permutation_test(
            xs=dialogue_report_df.loc[:, "prefix_0"].tolist(),
            ys=dialogue_report_df.loc[:, "prefix_1"].tolist(),
        )

        with open(scores_json, "w") as fp:
            fp.write(
                json.dumps(
                    {
                        "turn": {"statistic": turn_statistic, "pvalue": turn_pvalue},
                        "dialogue": {
                            "statistic": dialogue_statistic,
                            "pvalue": dialogue_pvalue,
                        },
                        "prefix": {"pvalue": prefix_pvalue},
                    },
                    indent=2,
                )
            )
            fp.write("\n")

    else:
        datum_ids = set(
            load_jsonl_file(data_jsonl=datum_ids_jsonl, cls=TurnId, verbose=False)
        )
        mask_datum_id = [
            TurnId(dialogue_id=dialogue_id, turn_index=turn_index) in datum_ids
            for (dialogue_id, turn_index), row in exp1_prediction_report_df.iterrows()
        ]
        turn_report_df = turn_report_df.loc[mask_datum_id]
        # NOTE: We only compute turn-level statistics since it doesn't make sense to compute dialogue-level metrics
        # with only a subset of turns.
        turn_statistic, turn_pvalue = run_mcnemar_test(turn_report_df)

        with open(scores_json, "w") as fp:
            fp.write(
                json.dumps(
                    {"turn": {"statistic": turn_statistic, "pvalue": turn_pvalue}},
                    indent=2,
                )
            )
            fp.write("\n")


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--exp0_prediction_report_tsv",
        help="the prediction report tsv file for one experiment exp0",
    )
    argument_parser.add_argument(
        "--exp1_prediction_report_tsv",
        help="the prediction report tsv file for the other experiment exp1",
    )
    argument_parser.add_argument(
        "--datum_ids_jsonl", default=None, help="if set, only evaluate on these turns",
    )
    argument_parser.add_argument("--scores_json", help="output scores json file")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        exp0_prediction_report_tsv=args.exp0_prediction_report_tsv,
        exp1_prediction_report_tsv=args.exp1_prediction_report_tsv,
        datum_ids_jsonl=args.datum_ids_jsonl,
        scores_json=args.scores_json,
    )
