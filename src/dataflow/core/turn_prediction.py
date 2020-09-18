#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from dataclasses import dataclass
from typing import Optional

from dataflow.core.dialogue import (
    Dialogue,
    ProgramExecutionOracle,
    TurnId,
    UserUtterance,
)


@dataclass(frozen=True, eq=True, repr=True)
class UtteranceWithContext:
    """
    A user utterance, with the dialogue history leading up to it.
    This is the input to the lispress prediction task.
    """

    datum_id: TurnId
    user_utterance: UserUtterance
    context: Dialogue


@dataclass(frozen=True, eq=True, repr=True)
class TurnPrediction:
    """
    A model prediction of the `lispress` for a single Turn.
    This is the output of the lispress prediction task.
    """

    datum_id: TurnId
    user_utterance: str  # redundant. just to make these files easier to read
    lispress: str


@dataclass(frozen=True, eq=True, repr=True)
class TurnAnswer:
    """
    A model prediction of the `lispress` for a single Turn.
    This is the output of the lispress prediction task.
    """

    datum_id: TurnId
    user_utterance: str  # redundant. just to make these files easier to read
    lispress: str
    program_execution_oracle: Optional[ProgramExecutionOracle]


def missing_prediction(datum_id: TurnId) -> TurnPrediction:
    """
    A padding `TurnPrediction` that is used when a turn with
    `datum_id` is missing from a predictions file.
    """
    return TurnPrediction(
        datum_id=datum_id, user_utterance="<missing>", lispress="<missing>",
    )
