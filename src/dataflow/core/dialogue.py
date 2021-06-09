#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from dataclasses import dataclass
from typing import List, Optional

from dataflow.core.linearize import lispress_to_seq
from dataflow.core.lispress import lispress_to_program, parse_lispress
from dataflow.core.program import Program


@dataclass(frozen=True)
class AgentUtterance:
    original_text: str
    tokens: List[str]
    # The ID of entities described in the agent utterance.
    # For programs that have inlined refer calls in our ablative study, the lispress at the
    # t-th turn would contain tokens that are entity IDs (e.g., "entity@12345") from
    # agent utterances in previous turns.
    # In order to make it possible for the seq2seq model to produce such tokens, we concatenate
    # these entity IDs in the source sequence side so that the model can learn to "copy" them into
    # the target sequence.
    # In normal programs with non-inlined refer calls, these entities would be retrieved through
    # the refer calls. Thus, we do not need to use this field for normal programs.
    described_entities: List[str]


@dataclass(frozen=True)
class UserUtterance:
    original_text: str
    tokens: List[str]


@dataclass(frozen=True)
class TurnId:
    dialogue_id: str
    turn_index: int

    def __hash__(self):
        return hash((self.dialogue_id, self.turn_index))


@dataclass(frozen=True)
class ProgramExecutionOracle:
    """The oracle information about the program execution.

    Because we have not implemented an executor in Python, we record the
    useful information about the execution results on the gold program annotations
    for evaluation purpose.
    """

    # the flag to indicate whether the program would raise any exception during execution
    has_exception: bool
    # the flag to indicate that whether all refer calls in the program would return the correct values during execution
    # NOTE: This flag is used in evaluation as if the refer calls are replaced with the concrete program
    # fragments they return. This means that a predicted plan is correct iff the plan itself matches the gold plan,
    # and the gold plan's refer calls are correct.
    refer_are_correct: bool


@dataclass(frozen=True)
class Turn:
    # the turn index
    turn_index: int
    # the current user utterance
    user_utterance: UserUtterance
    # the next agent utterance
    agent_utterance: AgentUtterance
    # the program corresponding to the user utterance
    # (see `dataflow.core.lispress` for the lisp string format)
    lispress: str
    # the flag to indicate whether to skip this turn when building the datum for training/prediction
    # Some turns are skipped to avoid the model from biasing to very common utterances, e.g., "yes", "okay".
    # NOTE: These turns are still used for building the dialog context even if the flag is true.
    skip: bool
    # the oracle information about the gold program
    program_execution_oracle: Optional[ProgramExecutionOracle] = None

    def tokenized_lispress(self) -> List[str]:
        return lispress_to_seq(parse_lispress(self.lispress))

    def program(self) -> Program:
        program, _ = lispress_to_program(parse_lispress(self.lispress), idx=0)
        return program


@dataclass(frozen=True)
class Dialogue:
    dialogue_id: str
    turns: List[Turn]
