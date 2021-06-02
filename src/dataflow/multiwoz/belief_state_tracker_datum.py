#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Slot:
    name: str
    value: str


@dataclass(frozen=True)
class BeliefState:
    slots_for_domain: Dict[str, List[Slot]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BeliefState):
            raise NotImplementedError()
        domains = set(self.slots_for_domain.keys())
        if domains != set(other.slots_for_domain.keys()):
            return False
        for domain in domains:
            if self.slots_for_domain[domain] != other.slots_for_domain[domain]:
                return False
        return True


@dataclass(frozen=True)
class BeliefStateTrackerDatum:
    """A datum for the belief state tracker.

    It is used as the universal data format for both gold and hypos in the
    evaluation script `evaluate_belief_state_predictions.py`.
    See factory methods in the script `create_belief_state_tracker_data.py`.
    """

    dialogue_id: str
    turn_index: int
    belief_state: BeliefState
    prev_agent_utterance: Optional[str] = None
    curr_user_utterance: Optional[str] = None


def pretty_print_belief_state(belief_state: BeliefState) -> str:
    return "\n".join(
        [
            "{}\t{}".format(
                domain,
                " | ".join(["{}={}".format(slot.name, slot.value) for slot in slots]),
            )
            for domain, slots in sorted(
                belief_state.slots_for_domain.items(), key=lambda x: x[0]
            )
        ]
    )


def sort_slots(slots_for_domain: Dict[str, List[Slot]]):
    """Sorts slots for each domain."""
    for domain in slots_for_domain:
        slots_for_domain[domain].sort(key=lambda x: (x.name, x.value))
