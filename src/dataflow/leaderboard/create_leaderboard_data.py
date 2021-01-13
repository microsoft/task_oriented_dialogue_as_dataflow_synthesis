#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Converts native Calflow data to the format used by the leaderboard.
"""
import argparse
import random
import string
from typing import List

from dataflow.core.dialogue import Dialogue, TurnId
from dataflow.core.io import save_jsonl_file, load_jsonl_file
from dataflow.core.turn_prediction import TurnAnswer, UtteranceWithContext


def get_random_string(length: int) -> str:
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for _ in range(length))
    return result_str


def main(
    dataflow_dialogues_jsonl: str,
    dialogue_id_prefix: str,
    contextualized_turns_file: str,
    turn_answers_file: str,
) -> None:
    new_dialogue_id_index = 0
    new_dialogue_ids = [get_random_string(16) for _ in range(500000)]
    new_dialogue_ids = list(set(new_dialogue_ids))
    contextualized_turns: List[UtteranceWithContext] = []
    turn_predictons: List[TurnAnswer] = []

    for dialogue in load_jsonl_file(data_jsonl=dataflow_dialogues_jsonl, cls=Dialogue, unit=" dialogues"):
        for turn_index, turn in enumerate(dialogue.turns):
            if turn.skip:
                continue
            full_dialogue_id = (
                dialogue_id_prefix + "-" + new_dialogue_ids[new_dialogue_id_index]
            )
            datum_id = TurnId(full_dialogue_id, turn.turn_index)
            contextualized_turn = UtteranceWithContext(
                datum_id=datum_id,
                user_utterance=turn.user_utterance,
                context=Dialogue(
                    dialogue_id=full_dialogue_id,
                    turns=dialogue.turns[:turn_index],
                ),
            )
            answer = TurnAnswer(
                datum_id=datum_id,
                user_utterance=turn.user_utterance.original_text,
                lispress=turn.lispress,
                program_execution_oracle=turn.program_execution_oracle,
            )
            contextualized_turns.append(contextualized_turn)
            turn_predictons.append(answer)
            new_dialogue_id_index += 1

    random.shuffle(contextualized_turns)
    save_jsonl_file(contextualized_turns, contextualized_turns_file)
    save_jsonl_file(turn_predictons, turn_answers_file)


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--dialogues_jsonl",
        help="the jsonl file containing the dialogue data with dataflow programs",
    )
    argument_parser.add_argument(
        "--contextualized_turns_file", help="the output file",
    )
    argument_parser.add_argument(
        "--turn_answers_file", help="the output file",
    )
    argument_parser.add_argument(
        "--dialogue_id_prefix", help="dialogue id prefix",
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
        dialogue_id_prefix=args.dialogue_id_prefix,
        contextualized_turns_file=args.contextualized_turns_file,
        turn_answers_file=args.turn_answers_file,
    )
