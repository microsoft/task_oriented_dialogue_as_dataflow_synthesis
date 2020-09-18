#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Creates text data (source-target pairs) to be used for training OpenNMT models.
"""
import argparse
import dataclasses
from typing import Dict, Iterator

import jsons
from tqdm import tqdm

from dataflow.core.dialogue import AgentUtterance, Turn
from dataflow.core.turn_prediction import UtteranceWithContext
from dataflow.onmt_helpers.create_onmt_text_data import (
    OnmtTextDatum,
    create_context_turns,
    create_onmt_text_datum_for_turn,
)

# We assume all dialogues start from turn 0.
# This is true for MultiWoZ and CalFlow datasets.
_MIN_TURN_INDEX = 0


def create_onmt_text_data_for_contextualized_turn(
    contextualized_turn: UtteranceWithContext,
    num_context_turns: int,
    min_turn_index: int,
    include_program: bool,
    include_agent_utterance: bool,
    include_described_entities: bool,
) -> Iterator[OnmtTextDatum]:
    """Yields OnmtTextDatum for a dialogue."""
    turn_lookup: Dict[int, Turn] = {
        turn.turn_index: turn for turn in contextualized_turn.context.turns
    }
    context_turns = create_context_turns(
        turn_lookup=turn_lookup,
        curr_turn_index=contextualized_turn.datum_id.turn_index,
        num_context_turns=num_context_turns,
        min_turn_index=min_turn_index,
    )
    onmt_text_datum = create_onmt_text_datum_for_turn(
        dialogue_id=contextualized_turn.datum_id.dialogue_id,
        curr_turn=Turn(
            turn_index=contextualized_turn.datum_id.turn_index,
            user_utterance=contextualized_turn.user_utterance,
            agent_utterance=AgentUtterance(
                original_text="", tokens=[], described_entities=[]
            ),
            lispress="()",
            skip=False,
        ),
        context_turns=context_turns,
        include_program=include_program,
        include_agent_utterance=include_agent_utterance,
        include_described_entities=include_described_entities,
    )
    yield onmt_text_datum


def main(
    dataflow_dialogues_jsonl: str,
    num_context_turns: int,
    min_turn_index: int,
    include_program: bool,
    include_agent_utterance: bool,
    include_described_entities: bool,
    onmt_text_data_outbase: str,
) -> None:
    fps = OnmtTextDatum.create_output_files(onmt_text_data_outbase)

    for line in tqdm(open(dataflow_dialogues_jsonl), unit=" contextualized turns"):
        contextualized_turn = jsons.loads(line.strip(), UtteranceWithContext)
        for onmt_text_datum in create_onmt_text_data_for_contextualized_turn(
            contextualized_turn=contextualized_turn,
            num_context_turns=num_context_turns,
            min_turn_index=min_turn_index,
            include_program=include_program,
            include_agent_utterance=include_agent_utterance,
            include_described_entities=include_described_entities,
        ):
            for field_name, field_value in dataclasses.asdict(onmt_text_datum).items():
                fp = fps[field_name]
                fp.write(field_value)
                fp.write("\n")

    for _, fp in fps.items():
        fp.close()


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--dialogues_jsonl",
        help="the jsonl file containing the dialogue data with dataflow programs",
    )
    argument_parser.add_argument(
        "--num_context_turns",
        type=int,
        help="number of previous turns to be included in the source sequence",
    )
    argument_parser.add_argument(
        "--include_program",
        default=False,
        action="store_true",
        help="if True, include the gold program for the context turn parts",
    )
    argument_parser.add_argument(
        "--include_agent_utterance",
        default=False,
        action="store_true",
        help="if True, include the gold agent utterance for the context turn parts",
    )
    argument_parser.add_argument(
        "--include_described_entities",
        default=False,
        action="store_true",
        help="if True, include the described entities field for the context turn parts",
    )
    argument_parser.add_argument(
        "--onmt_text_data_outbase",
        help="the output file basename for the extracted text data for OpenNMT",
    )


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        dataflow_dialogues_jsonl=args.dialogues_jsonl,
        num_context_turns=args.num_context_turns,
        min_turn_index=_MIN_TURN_INDEX,
        include_program=args.include_program,
        include_agent_utterance=args.include_agent_utterance,
        include_described_entities=args.include_described_entities,
        onmt_text_data_outbase=args.onmt_text_data_outbase,
    )
