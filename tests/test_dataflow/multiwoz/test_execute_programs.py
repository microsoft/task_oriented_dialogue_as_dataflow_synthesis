#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from typing import Any, Dict

from test_dataflow.multiwoz.conftest import build_trade_dialogue

from dataflow.core.utterance_tokenizer import UtteranceTokenizer
from dataflow.multiwoz.create_belief_state_tracker_data import (
    build_belief_state_from_belief_dict,
    build_belief_state_from_trade_turn,
)
from dataflow.multiwoz.create_programs import create_programs_for_trade_dialogue
from dataflow.multiwoz.execute_programs import execute_programs_for_dialogue
from dataflow.multiwoz.salience_model import VanillaSalienceModel


def test_execute_programs(trade_dialogue_1: Dict[str, Any]):
    utterance_tokenizer = UtteranceTokenizer()
    salience_model = VanillaSalienceModel()

    # ============================
    # get cheating execution results
    # ============================
    dataflow_dialogue, _, _ = create_programs_for_trade_dialogue(
        trade_dialogue=trade_dialogue_1,
        keep_all_domains=True,
        remove_none=False,
        fill_none=False,
        salience_model=salience_model,
        no_revise=False,
        avoid_empty_plan=False,
        utterance_tokenizer=utterance_tokenizer,
    )
    complete_execution_results, cheating_turn_indices = execute_programs_for_dialogue(
        dialogue=dataflow_dialogue,
        salience_model=salience_model,
        no_revise=False,
        cheating_mode="never",
        cheating_execution_results=None,
    )
    assert not cheating_turn_indices
    for trade_turn, complete_execution_result in zip(
        trade_dialogue_1["dialogue"], complete_execution_results
    ):
        assert build_belief_state_from_trade_turn(
            trade_turn
        ) == build_belief_state_from_belief_dict(
            complete_execution_result.belief_dict, strict=True
        )
    # pylint: disable=no-member
    cheating_execution_results = {
        turn.turn_index: complete_execution_result
        for turn, complete_execution_result in zip(
            dataflow_dialogue.turns, complete_execution_results
        )
    }

    # ============================
    # mock the belief state predictions
    # ============================
    mock_belief_states = [
        # turn 1: correct
        {"hotel-name": "none", "hotel-type": "none"},
        # turn 2: correct
        {
            "hotel-name": "hilton",
            "hotel-pricerange": "cheap",
            "hotel-type": "guest house",
        },
        # turn 3: change a slot
        {
            "hotel-name": "none",
            "hotel-pricerange": "_cheap",
            "hotel-type": "guest house",
        },
        # turn 4: add a slot
        {"hotel-type": "_added"},
        # turn 5: correct
        {"hotel-area": "west"},
        # turn 6: correct
        {"hotel-area": "west", "restaurant-area": "west"},
        # turn 7: correct
        {
            "hotel-area": "west",
            "restaurant-area": "west",
            "restaurant-pricerange": "cheap",
        },
        # turn 8: change two slots
        {
            "hotel-area": "_west",
            "restaurant-area": "_west",
            "restaurant-pricerange": "cheap",
        },
        # turn 9: correct
        {
            "hotel-area": "west",
            "restaurant-area": "west",
            "restaurant-pricerange": "cheap",
            "taxi-departure": "none",
        },
        # turn 10: drop a slot
        {
            "hotel-area": "west",
            "restaurant-area": "west",
            "restaurant-pricerange": "cheap",
        },
    ]
    mock_trade_dialogue = build_trade_dialogue(
        dialogue_id="mock",
        turns=[("", "", belief_state) for belief_state in mock_belief_states],
    )

    mock_dataflow_dialogue, _, _ = create_programs_for_trade_dialogue(
        trade_dialogue=mock_trade_dialogue,
        keep_all_domains=True,
        remove_none=False,
        fill_none=False,
        salience_model=salience_model,
        no_revise=False,
        avoid_empty_plan=False,
        utterance_tokenizer=utterance_tokenizer,
    )
    _, mock_cheating_turn_indices = execute_programs_for_dialogue(
        dialogue=mock_dataflow_dialogue,
        salience_model=salience_model,
        no_revise=False,
        cheating_mode="always",
        cheating_execution_results=cheating_execution_results,
    )
    assert mock_cheating_turn_indices == [
        turn.turn_index for turn in dataflow_dialogue.turns
    ]

    _, mock_cheating_turn_indices = execute_programs_for_dialogue(
        dialogue=mock_dataflow_dialogue,
        salience_model=salience_model,
        no_revise=False,
        cheating_mode="dynamic",
        cheating_execution_results=cheating_execution_results,
    )
    assert mock_cheating_turn_indices == [3, 4, 8, 10]
