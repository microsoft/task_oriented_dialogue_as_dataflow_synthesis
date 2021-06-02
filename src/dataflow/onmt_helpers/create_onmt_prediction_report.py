#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Creates the prediction report from onmt_translate output.
"""
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterator, List, Union

import jsons
from more_itertools import chunked

from dataflow.core.dialogue import (
    AgentUtterance,
    Dialogue,
    ProgramExecutionOracle,
    Turn,
    TurnId,
    UserUtterance,
)
from dataflow.core.io import (
    load_jsonl_file,
    load_jsonl_file_and_build_lookup,
    save_jsonl_file,
)
from dataflow.core.linearize import seq_to_lispress, to_canonical_form
from dataflow.core.lispress import render_compact
from dataflow.core.prediction_report import (
    PredictionReportDatum,
    save_prediction_report_tsv,
    save_prediction_report_txt,
)

_DUMMY_USER_UTTERANCE = UserUtterance(original_text="", tokens=[])
_DUMMY_AGENT_UTTERANCE = AgentUtterance(
    original_text="", tokens=[], described_entities=[]
)
_PARSE_ERROR_LISPRESS = '(parseError #(InvalidLispress "")'


@dataclass(frozen=True)
class OnmtPredictionReportDatum(PredictionReportDatum):
    datum_id: TurnId
    source: str
    # The tokenized gold lispress.
    gold: str
    # The tokenized predicted lispress.
    prediction: str
    program_execution_oracle: ProgramExecutionOracle

    @property
    def gold_canonical(self) -> str:
        return to_canonical_form(self.gold)

    @property
    def prediction_canonical(self) -> str:
        try:
            return to_canonical_form(self.prediction)
        except Exception:  # pylint: disable=W0703
            return _PARSE_ERROR_LISPRESS

    @property
    def is_correct(self) -> bool:
        return (
            self.gold == self.prediction
            and self.program_execution_oracle.refer_are_correct
        )

    @property
    def is_correct_leaderboard(self) -> bool:
        """Returns true if the gold and the prediction match after canonicalization.

        This is the metric used in the leaderboard, which would be slightly higher than the one reported in the TACL2020
        paper, since the named arguments are sorted after canonicalization.
        """
        return (
            self.gold_canonical == self.prediction_canonical
            and self.program_execution_oracle.refer_are_correct
        )

    def flatten_datum_id(self) -> Dict[str, Union[str, int]]:
        return {
            "dialogueId": self.datum_id.dialogue_id,
            "turnIndex": self.datum_id.turn_index,
        }

    def flatten(self) -> Dict[str, Union[str, int]]:
        flatten_datum_dict = self.flatten_datum_id()
        # It's fine to call update since we always return a new dict from self.flatten_datum_id().
        flatten_datum_dict.update(
            {
                "source": self.source,
                "gold": self.gold,
                "prediction": self.prediction,
                "goldCanonical": self.gold_canonical,
                "predictionCanonical": self.prediction_canonical,
                "oracleResolveAreCorrect": self.program_execution_oracle.refer_are_correct,
                "isCorrect": self.is_correct,
                "isCorrectLeaderboard": self.is_correct_leaderboard,
            }
        )
        return flatten_datum_dict


def build_prediction_report_datum(
    datum_lookup: Dict[str, Dict[int, Turn]],
    datum_id_line: str,
    src_line: str,
    ref_line: str,
    nbest_lines: List[str],
) -> OnmtPredictionReportDatum:
    datum_id = jsons.loads(datum_id_line.strip(), TurnId)
    datum = datum_lookup[datum_id.dialogue_id][datum_id.turn_index]
    return OnmtPredictionReportDatum(
        datum_id=datum_id,
        source=src_line.strip(),
        gold=ref_line.strip(),
        prediction=nbest_lines[0].strip(),
        program_execution_oracle=datum.program_execution_oracle,
    )


def create_onmt_prediction_report(
    datum_lookup: Dict[str, Dict[int, Turn]],
    datum_id_jsonl: str,
    src_txt: str,
    ref_txt: str,
    nbest_txt: str,
    nbest: int,
    outbase: str,
) -> str:
    prediction_report = [
        build_prediction_report_datum(
            datum_lookup=datum_lookup,
            datum_id_line=datum_id_line,
            src_line=src_line,
            ref_line=ref_line,
            nbest_lines=nbest_lines,
        )
        for datum_id_line, src_line, ref_line, nbest_lines in zip(
            open(datum_id_jsonl),
            open(src_txt),
            open(ref_txt),
            chunked(open(nbest_txt), nbest),
        )
    ]
    prediction_report.sort(key=lambda x: dataclasses.astuple(x.datum_id))
    predictions_jsonl = f"{outbase}.prediction_report.jsonl"
    save_jsonl_file(prediction_report, predictions_jsonl)
    save_prediction_report_tsv(prediction_report, f"{outbase}.prediction_report.tsv")
    save_prediction_report_txt(
        prediction_report=prediction_report,
        prediction_report_txt=f"{outbase}.prediction_report.txt",
        field_names=[
            "dialogueId",
            "turnIndex",
            "source",
            "oracleResolveAreCorrect",
            "isCorrect",
            "isCorrectLeaderboard",
            "gold",
            "prediction",
            "goldCanonical",
            "predictionCanonical",
        ],
    )
    return predictions_jsonl


def build_dataflow_dialogue(
    dialogue_id: str, prediction_report_data: Dict[int, OnmtPredictionReportDatum]
) -> Dialogue:
    turns: List[Turn] = []
    datum: OnmtPredictionReportDatum
    for turn_index, datum in sorted(prediction_report_data.items(), key=lambda x: x[0]):
        # pylint: disable=broad-except
        tokenized_lispress = datum.prediction.split(" ")
        try:
            lispress = render_compact(seq_to_lispress(tokenized_lispress))
        except Exception as e:
            print(e)
            lispress = _PARSE_ERROR_LISPRESS

        turns.append(
            Turn(
                turn_index=turn_index,
                user_utterance=_DUMMY_USER_UTTERANCE,
                agent_utterance=_DUMMY_AGENT_UTTERANCE,
                lispress=lispress,
                skip=False,
                program_execution_oracle=None,
            )
        )

    return Dialogue(dialogue_id=dialogue_id, turns=turns)


def build_dataflow_dialogues(
    prediction_report_data_lookup: Dict[str, Dict[int, OnmtPredictionReportDatum]]
) -> Iterator[Dialogue]:
    for dialogue_id, prediction_report_data in prediction_report_data_lookup.items():
        dataflow_dialogue = build_dataflow_dialogue(
            dialogue_id=dialogue_id, prediction_report_data=prediction_report_data
        )
        yield dataflow_dialogue


def main(
    dialogues_jsonl: str,
    datum_id_jsonl: str,
    src_txt: str,
    ref_txt: str,
    nbest_txt: str,
    nbest: int,
    outbase: str,
) -> None:
    """Creates 1-best predictions and saves them to files."""
    datum_lookup: Dict[str, Dict[int, Turn]] = {
        dialogue.dialogue_id: {turn.turn_index: turn for turn in dialogue.turns}
        for dialogue in load_jsonl_file(
            data_jsonl=dialogues_jsonl, cls=Dialogue, unit=" dialogues"
        )
    }

    prediction_report_jsonl = create_onmt_prediction_report(
        datum_lookup=datum_lookup,
        datum_id_jsonl=datum_id_jsonl,
        src_txt=src_txt,
        ref_txt=ref_txt,
        nbest_txt=nbest_txt,
        nbest=nbest,
        outbase=outbase,
    )

    predictions_lookup = load_jsonl_file_and_build_lookup(
        data_jsonl=prediction_report_jsonl,
        cls=OnmtPredictionReportDatum,
        primary_key_getter=lambda x: x.datum_id.dialogue_id,
        secondary_key_getter=lambda x: x.datum_id.turn_index,
    )
    dataflow_dialogues = build_dataflow_dialogues(predictions_lookup)
    save_jsonl_file(dataflow_dialogues, f"{outbase}.dataflow_dialogues.jsonl")


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--dialogues_jsonl",
        help="the jsonl file containing the dialogue data with dataflow programs",
    )
    argument_parser.add_argument("--datum_id_jsonl", help="datum ID file")
    argument_parser.add_argument("--src_txt", help="source sequence file")
    argument_parser.add_argument("--ref_txt", help="target sequence reference file")
    argument_parser.add_argument("--nbest_txt", help="onmt_translate output file")
    argument_parser.add_argument("--nbest", type=int, help="number of hypos per datum")
    argument_parser.add_argument("--outbase", help="the basename of output files")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        dialogues_jsonl=args.dialogues_jsonl,
        datum_id_jsonl=args.datum_id_jsonl,
        src_txt=args.src_txt,
        ref_txt=args.ref_txt,
        nbest_txt=args.nbest_txt,
        nbest=args.nbest,
        outbase=args.outbase,
    )
