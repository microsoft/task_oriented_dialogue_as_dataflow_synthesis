#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class PartialExecutionResult:
    """A partial execution result for an express at a turn.
    """

    # the values in the dataflow graph
    # - key: expression ID
    # - value: the underlying value, None means removing the slot from the belief state
    values: Dict[str, Optional[str]]
    # the constraints in the dataflow graph
    constraints: Dict[str, str]
    # the refer call logs
    # - key: the expression ID that uses the refer call
    # - value: the target type
    refer_calls: Dict[str, str]
    # the typed slot values
    # - key: the slot name (without the domain)
    # - value: the history of (slotValue, expressionId) for this type (slotValue==None means deletion)
    slot_values: Dict[str, List[Tuple[Optional[str], Optional[str]]]]


@dataclass(frozen=True)
class ExecutionTrace:
    """A complete execution trace for a dialogue up to a turn."""

    # - key: slot name without the domain
    # - value: the history of (slotValue, expressionId) for this type (slotValue==None means deletion)
    slot_values: Dict[str, List[Tuple[Optional[str], Optional[str]]]]


class SalienceModelBase(ABC):
    """An abstract class for the salience model.
    """

    @abstractmethod
    def get_salient_value(
        self,
        target_type: str,
        execution_trace: ExecutionTrace,
        exclude_values: Set[str],
    ) -> Optional[str]:
        """Gets the salient mention from dialogue context.

        Args:
            target_type: the target type of the salient value
            execution_trace: the execution_trace up to the previous turn
            exclude_values: values should not be returned
        Returns:
            The retrieved salient value for the target type.
        """
        raise NotImplementedError()


class DummySalienceModel(SalienceModelBase):
    """A dummy salience model which always returns None.
    """

    def get_salient_value(
        self,
        target_type: str,
        execution_trace: ExecutionTrace,
        exclude_values: Set[str],
    ) -> Optional[str]:
        """See base class.

        For a dummy salience model, we always return None.
        """
        return None


class VanillaSalienceModel(SalienceModelBase):
    """A vanilla salience model.
    """

    # the ontology for slot types
    # it records the compatible slot names for salience calls
    # the lookup from slot name to slot type
    SLOT_TYPE_ONTOLOGY: Dict[str, List[str]] = {
        "PLACE": ["name", "destination", "departure"],
        "DAY": ["day", "book-day"],
        "TIME": ["book-time", "arriveby", "leaveat"],
        "NUMBER": ["book-people", "stars", "book-stay"],
    }
    SLOT_TYPE_LOOKUP: Dict[str, str] = {
        slot_name: slot_type
        for slot_type, slot_names in SLOT_TYPE_ONTOLOGY.items()
        for slot_name in slot_names
    }
    SLOT_VALUE_BLOCKLIST: Set[Optional[str]] = {"none", "dontcare", None}

    def get_salient_value(
        self,
        target_type: str,
        execution_trace: ExecutionTrace,
        exclude_values: Set[str],
    ) -> Optional[str]:
        """See base class.

        Currently, this method returns the most recent occurrence of the value that is compatible with the target type.
        """
        for value, _ in reversed(execution_trace.slot_values.get(target_type, [])):
            if value in exclude_values:
                continue
            if value in self.SLOT_VALUE_BLOCKLIST:
                continue
            return value

        slot_type = self.SLOT_TYPE_LOOKUP.get(target_type, None)
        if slot_type is not None:
            for slot_name in self.SLOT_TYPE_ONTOLOGY[slot_type]:
                for value, _ in reversed(
                    execution_trace.slot_values.get(slot_name, [])
                ):
                    if value in exclude_values:
                        continue
                    if value in self.SLOT_VALUE_BLOCKLIST:
                        continue
                    return value

        return None
