#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Compute statistics for the dataflow dialogues.
"""

import argparse
import dataclasses
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dataflow.core.dialogue import Dialogue, Turn, TurnId
from dataflow.core.io import load_jsonl_file, save_jsonl_file
from dataflow.core.program_utils import DataflowFn


@dataclass(frozen=True)
class BasicStatistics:
    num_dialogues: int
    num_turns: int
    num_kept_turns: int
    num_skipped_turns: int
    num_refer_turns: int
    num_revise_turns: int


def is_refer_turn(turn: Turn) -> bool:
    if re.search(rf"\({DataflowFn.Refer.value} ", turn.lispress):
        return True
    return False


def is_revise_turn(turn: Turn) -> bool:
    if re.search(rf"\({DataflowFn.Revise.value} ", turn.lispress):
        return True
    return False


def build_dialogue_report(
    dataflow_dialogues: List[Dialogue],
) -> Tuple[pd.DataFrame, List[TurnId], List[TurnId]]:
    refer_turn_ids = []
    revise_turn_ids = []
    report_rows = []

    for dialogue in dataflow_dialogues:
        num_turns = len(dialogue.turns)
        num_kept_turns = 0
        num_skipped_turns = 0
        num_refer_turns = 0
        num_revise_turns = 0
        for turn in dialogue.turns:
            if turn.skip:
                num_skipped_turns += 1
                continue

            num_kept_turns += 1
            if is_refer_turn(turn):
                num_refer_turns += 1
                refer_turn_ids.append(
                    TurnId(dialogue_id=dialogue.dialogue_id, turn_index=turn.turn_index)
                )
            if is_revise_turn(turn):
                num_revise_turns += 1
                revise_turn_ids.append(
                    TurnId(dialogue_id=dialogue.dialogue_id, turn_index=turn.turn_index)
                )

        report_rows.append(
            {
                "dialogueId": dialogue.dialogue_id,
                "numTurns": num_turns,
                "numKeptTurns": num_kept_turns,
                "numSkippedTurns": num_skipped_turns,
                "numReferTurns": num_refer_turns,
                "numReviseTurns": num_revise_turns,
            }
        )

    report_df = pd.DataFrame(report_rows)
    return report_df, refer_turn_ids, revise_turn_ids


def compute_stats(
    dialogue_report_df: pd.DataFrame,
) -> Tuple[BasicStatistics, Dict[str, List[float]]]:
    basic_stats = BasicStatistics(
        num_dialogues=len(dialogue_report_df),
        num_turns=int(dialogue_report_df.loc[:, "numTurns"].sum()),
        num_kept_turns=int(dialogue_report_df.loc[:, "numKeptTurns"].sum()),
        num_skipped_turns=int(dialogue_report_df.loc[:, "numSkippedTurns"].sum()),
        num_refer_turns=int(dialogue_report_df.loc[:, "numReferTurns"].sum()),
        num_revise_turns=int(dialogue_report_df.loc[:, "numReviseTurns"].sum()),
    )
    percentiles = list(range(0, 101, 10))
    percentile_stats = {
        field: list(
            np.percentile(dialogue_report_df.loc[:, field].tolist(), percentiles)
        )
        for field in [
            "numTurns",
            "numKeptTurns",
            "numSkippedTurns",
            "numReferTurns",
            "numReviseTurns",
        ]
    }

    return basic_stats, percentile_stats


def main(dataflow_dialogues_dir: str, subsets: List[str], outdir: str):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    dialogue_report_dfs = []
    for subset in subsets:
        dataflow_dialogues = list(
            load_jsonl_file(
                data_jsonl=os.path.join(
                    dataflow_dialogues_dir, f"{subset}.dataflow_dialogues.jsonl"
                ),
                cls=Dialogue,
                unit=" dialogues",
            )
        )

        dialogue_report_df, refer_turn_ids, revise_turn_ids = build_dialogue_report(
            dataflow_dialogues
        )
        dialogue_report_dfs.append(dialogue_report_df)

        save_jsonl_file(
            data=refer_turn_ids,
            data_jsonl=os.path.join(outdir, f"{subset}.refer_turn_ids.jsonl"),
        )
        save_jsonl_file(
            data=revise_turn_ids,
            data_jsonl=os.path.join(outdir, f"{subset}.revise_turn_ids.jsonl"),
        )

        basic_stats, percentile_stats = compute_stats(dialogue_report_df)
        with open(os.path.join(outdir, f"{subset}.basic_stats.json"), "w") as fp:
            fp.write(json.dumps(dataclasses.asdict(basic_stats), indent=2))
            fp.write("\n")
        with open(os.path.join(outdir, f"{subset}.percentile_stats.json"), "w") as fp:
            fp.write(json.dumps(percentile_stats, indent=2))
            fp.write("\n")

    if len(subsets) > 1:
        basic_stats, percentile_stats = compute_stats(pd.concat(dialogue_report_dfs))
        with open(
            os.path.join(outdir, f"{'-'.join(subsets)}.basic_stats.json"), "w"
        ) as fp:
            fp.write(json.dumps(dataclasses.asdict(basic_stats), indent=2))
            fp.write("\n")
        with open(
            os.path.join(outdir, f"{'-'.join(subsets)}.percentile_stats.json"), "w"
        ) as fp:
            fp.write(json.dumps(percentile_stats, indent=2))
            fp.write("\n")


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--dataflow_dialogues_dir", help="the dataflow dialogues data directory"
    )
    argument_parser.add_argument(
        "--subset", nargs="+", default=[], help="the subset to be analyzed"
    )
    argument_parser.add_argument("--outdir", help="the output directory")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        dataflow_dialogues_dir=args.dataflow_dialogues_dir,
        subsets=args.subset,
        outdir=args.outdir,
    )
