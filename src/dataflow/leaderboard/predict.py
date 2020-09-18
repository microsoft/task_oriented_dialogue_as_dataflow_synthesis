#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Creates the prediction files from onmt_translate output for the leaderboard.
"""
import argparse
from typing import List

import jsons
from more_itertools import chunked

from dataflow.core.dialogue import TurnId
from dataflow.core.io import save_jsonl_file
from dataflow.core.turn_prediction import TurnPrediction


def build_prediction_report_datum(
    datum_id_line: str, src_line: str, nbest_lines: List[str],
) -> TurnPrediction:
    datum_id = jsons.loads(datum_id_line.strip(), TurnId)
    return TurnPrediction(
        datum_id=datum_id,
        user_utterance=src_line.strip(),
        lispress=nbest_lines[0].strip(),
    )


def create_onmt_prediction_report(
    datum_id_jsonl: str, src_txt: str, ref_txt: str, nbest_txt: str, nbest: int,
):
    prediction_report = [
        build_prediction_report_datum(
            datum_id_line=datum_id_line, src_line=src_line, nbest_lines=nbest_lines,
        )
        for datum_id_line, src_line, ref_line, nbest_lines in zip(
            open(datum_id_jsonl),
            open(src_txt),
            open(ref_txt),
            chunked(open(nbest_txt), nbest),
        )
    ]
    save_jsonl_file(prediction_report, "predictions.jsonl")


def main(
    datum_id_jsonl: str, src_txt: str, ref_txt: str, nbest_txt: str, nbest: int,
) -> None:
    """Creates 1-best predictions and saves them to files."""
    create_onmt_prediction_report(
        datum_id_jsonl=datum_id_jsonl,
        src_txt=src_txt,
        ref_txt=ref_txt,
        nbest_txt=nbest_txt,
        nbest=nbest,
    )


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument("--datum_id_jsonl", help="datum ID file")
    argument_parser.add_argument("--src_txt", help="source sequence file")
    argument_parser.add_argument("--ref_txt", help="target sequence reference file")
    argument_parser.add_argument("--nbest_txt", help="onmt_translate output file")
    argument_parser.add_argument("--nbest", type=int, help="number of hypos per datum")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        datum_id_jsonl=args.datum_id_jsonl,
        src_txt=args.src_txt,
        ref_txt=args.ref_txt,
        nbest_txt=args.nbest_txt,
        nbest=args.nbest,
    )
