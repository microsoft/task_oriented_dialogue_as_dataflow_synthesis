#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Creates text data (source-target pairs) to be used for training OpenNMT models."""
import argparse
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, TextIO

import jsons
from tqdm import tqdm

from dataflow.core.constants import SpecialStrings
from dataflow.core.dialogue import Dialogue, Turn, TurnId

# We assume all dialogues start from turn 0.
# This is true for MultiWoZ and CalFlow datasets.
_MIN_TURN_INDEX = 0


@dataclass(frozen=True)
class OnmtTextDatum:
    # datum ID
    datum_id_str: str
    # source string (raw)
    src_str: str
    # tokenized source string
    src_tok_str: str
    # target string
    tgt_str: str

    @classmethod
    def create_output_files(cls, outbase: str) -> Dict[str, TextIO]:
        return {
            f"{suffix}_str": open("{}.{}".format(outbase, suffix), "w")
            for suffix in ["datum_id", "src", "src_tok", "tgt"]
        }


def stringify_turn(
    turn: Turn,
    include_user_utterance: bool,
    include_program: bool,
    include_agent_utterance: bool,
    include_described_entities: bool,
    tokenize_utterance: bool,
) -> str:
    """Render a turn as a string and inserts corresponding delimiters before each segment in the string."""
    segments: List[str] = []

    if include_user_utterance:
        segments.append(SpecialStrings.SPEAKER_USER)
        if tokenize_utterance:
            assert turn.user_utterance.tokens
            user_utterance_str = " ".join(turn.user_utterance.tokens)
        else:
            user_utterance_str = turn.user_utterance.original_text
        segments.append(user_utterance_str)

    if include_program:
        segments.append(SpecialStrings.START_OF_PROGRAM)
        lispress_tokens = turn.tokenized_lispress()
        assert lispress_tokens
        segments.append(" ".join(lispress_tokens))

    if include_agent_utterance:
        segments.append(SpecialStrings.SPEAKER_AGENT)
        if tokenize_utterance:
            assert turn.agent_utterance.tokens
            agent_utterance_str = " ".join(turn.agent_utterance.tokens)
        else:
            agent_utterance_str = turn.agent_utterance.original_text
        segments.append(agent_utterance_str)

    if include_described_entities and turn.agent_utterance.described_entities:
        # NOTE: Should check if `described_entities` is empty or not. Otherwise, this will
        # introduce an empty token. This is fine for OpenNMT though because it uses `string.split()`
        # to tokenize the string (See the `tokenizer` attribute in `torchtext.data.field.Field`).
        segments.append(" ".join(turn.agent_utterance.described_entities))

    assert segments

    return " ".join(segments)


def create_source_str(
    curr_turn: Turn,
    context_turns: List[Turn],
    include_program: bool,
    include_agent_utterance: bool,
    include_described_entities: bool,
    tokenize_utterance: bool,
) -> str:
    """Creates the source sequence string."""
    segments: List[str] = []
    # context turn parts (may be empty)
    segments += [
        stringify_turn(
            turn=context_turn,
            include_user_utterance=True,
            include_program=include_program,
            include_agent_utterance=include_agent_utterance,
            include_described_entities=include_described_entities,
            tokenize_utterance=tokenize_utterance,
        )
        for idx, context_turn in enumerate(context_turns)
    ]
    # user utterance for the current turn part
    # - include the user utterance
    # - do not include the gold program
    # - do not include the gold agent utterance
    segments += [
        stringify_turn(
            turn=curr_turn,
            include_user_utterance=True,
            include_program=False,
            include_agent_utterance=False,
            include_described_entities=False,
            tokenize_utterance=tokenize_utterance,
        ),
        # add this special token to trigger the decoder to produce the program
        SpecialStrings.START_OF_PROGRAM,
    ]
    return " ".join(segments)


def create_onmt_text_datum_for_turn(
    dialogue_id: str,
    curr_turn: Turn,
    context_turns: List[Turn],
    include_program: bool,
    include_agent_utterance: bool,
    include_described_entities: bool,
) -> OnmtTextDatum:
    """Creates the OpenNMT text datum for a turn."""
    datum_id_str = jsons.dumps(TurnId(dialogue_id, curr_turn.turn_index))
    src_str = create_source_str(
        curr_turn=curr_turn,
        context_turns=context_turns,
        include_program=include_program,
        include_agent_utterance=include_agent_utterance,
        include_described_entities=include_described_entities,
        tokenize_utterance=False,
    )
    src_tok_str = create_source_str(
        curr_turn=curr_turn,
        context_turns=context_turns,
        include_program=include_program,
        include_agent_utterance=include_agent_utterance,
        include_described_entities=include_described_entities,
        tokenize_utterance=True,
    )
    tgt_str = " ".join(curr_turn.tokenized_lispress())

    # make sure there are not consecutive spaces in the tokenized sequence
    assert re.search(r"\s{2,}", src_tok_str) is None
    assert re.search(r"\s{2,}", tgt_str) is None

    return OnmtTextDatum(
        datum_id_str=datum_id_str,
        src_str=src_str,
        src_tok_str=src_tok_str,
        tgt_str=tgt_str,
    )


def create_context_turns(
    turn_lookup: Dict[int, Turn],
    curr_turn_index: int,
    num_context_turns: int,
    min_turn_index: int,
) -> List[Turn]:
    return [
        turn_lookup[tt]
        for tt in range(
            max(min_turn_index, curr_turn_index - num_context_turns), curr_turn_index
        )
    ]


def create_onmt_text_data_for_dialogue(
    dialogue: Dialogue,
    num_context_turns: int,
    min_turn_index: int,
    include_program: bool,
    include_agent_utterance: bool,
    include_described_entities: bool,
) -> Iterator[OnmtTextDatum]:
    """Yields OnmtTextDatum for a dialogue."""
    turn_lookup: Dict[int, Turn] = {turn.turn_index: turn for turn in dialogue.turns}
    for turn_index, turn in turn_lookup.items():
        if turn.skip:
            continue

        context_turns = create_context_turns(
            turn_lookup=turn_lookup,
            curr_turn_index=turn_index,
            num_context_turns=num_context_turns,
            min_turn_index=min_turn_index,
        )
        onmt_text_datum = create_onmt_text_datum_for_turn(
            dialogue_id=dialogue.dialogue_id,
            curr_turn=turn,
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

    for line in tqdm(open(dataflow_dialogues_jsonl), unit=" dialogues"):
        dialogue: Dialogue
        dialogue = jsons.loads(line.strip(), Dialogue)

        for onmt_text_datum in create_onmt_text_data_for_dialogue(
            dialogue=dialogue,
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
