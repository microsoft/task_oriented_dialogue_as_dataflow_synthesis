#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from typing import Any, Dict, List, Tuple

import pytest

from dataflow.multiwoz.trade_dst_utils import BeliefState


def convert_belief_dict_to_belief_state(belief_dict: Dict[str, str]) -> BeliefState:
    belief_state: BeliefState = []
    for slot_fullname, slot_value in sorted(belief_dict.items()):
        belief_state.append({"slots": [[slot_fullname, slot_value]]})
    return belief_state


def build_trade_dialogue(
    dialogue_id: str, turns: List[Tuple[str, str, Dict[str, str]]]
) -> Dict[str, Any]:
    trade_dialogue = {
        "dialogue_idx": dialogue_id,
        "dialogue": [
            {
                # Our mock dialogues here use 1-based turn indices.
                # In real MultiWOZ/TRADE dialogues, turn index starts from 0.
                "turn_idx": turn_idx + 1,
                "system_transcript": agent_utt,
                "transcript": user_utt,
                "belief_state": convert_belief_dict_to_belief_state(belief_dict),
            }
            for turn_idx, (agent_utt, user_utt, belief_dict) in enumerate(turns)
        ],
    }
    return trade_dialogue


@pytest.fixture
def trade_dialogue_1() -> Dict[str, Any]:
    return build_trade_dialogue(
        dialogue_id="dummy_1",
        turns=[
            # turn 1
            # activate a domain without constraint, the plan should call "Find" with "EqualityConstraint"
            # we intentionally to only put two "none" slots in the belief state to match the MultiWoZ annotation style
            (
                "",
                "i want to book a hotel",
                {"hotel-name": "none", "hotel-type": "none"},
            ),
            # turn 2
            # add constraints, the plan should call "Revise" with "EqualityConstraint"
            (
                "ok what type",
                "guest house and cheap, probably hilton",
                {
                    "hotel-name": "hilton",
                    "hotel-pricerange": "cheap",
                    "hotel-type": "guest house",
                },
            ),
            # turn 3
            # drop a constraint (but the domain is still active), the plan should call "Revise" with "EqualityConstraint"
            (
                "no results",
                "ok try another hotel",
                {
                    "hotel-name": "none",
                    "hotel-pricerange": "cheap",
                    "hotel-type": "guest house",
                },
            ),
            # turn 4
            # drop the domain
            ("failed", "ok never mind", {}),
            # turn 5
            # activate the domain again
            ("sure", "can you find a hotel in west", {"hotel-area": "west"}),
            # turn 6
            # activate a new domain and use a refer call
            (
                "how about this",
                "ok can you find a restaurant in the same area",
                {"hotel-area": "west", "restaurant-area": "west"},
            ),
            # turn 7
            # use a refer call to get a value from a dead domain
            # the salience model should find the first valid refer value (skips "none")
            (
                "how about this",
                "use the same price range as the hotel",
                {
                    "hotel-area": "west",
                    "restaurant-area": "west",
                    "restaurant-pricerange": "cheap",
                },
            ),
            # turn 8
            # do not change belief state
            (
                "ok",
                "give me the address",
                {
                    "hotel-area": "west",
                    "restaurant-area": "west",
                    "restaurant-pricerange": "cheap",
                },
            ),
            # turn 9
            # a new domain
            (
                "ok",
                "book a taxi now",
                {
                    "hotel-area": "west",
                    "restaurant-area": "west",
                    "restaurant-pricerange": "cheap",
                    "taxi-departure": "none",
                },
            ),
            # turn 10
            # do not change belief state (make sure the plan is "Revise" not "Find")
            (
                "ok",
                "ok",
                {
                    "hotel-area": "west",
                    "restaurant-area": "west",
                    "restaurant-pricerange": "cheap",
                    "taxi-departure": "none",
                },
            ),
        ],
    )
