#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import json
import os
from typing import Any, Dict, Iterator, List

from dataflow.core.utterance_tokenizer import UtteranceTokenizer
from dataflow.multiwoz.create_programs import create_programs_for_trade_dialogue
from dataflow.multiwoz.salience_model import DummySalienceModel, VanillaSalienceModel


def load_test_trade_dialogues(data_dir: str) -> Iterator[Dict[str, Any]]:
    """Returns selected test dialogues.

    To extract a dialogue from the TRADE processed json file:
    $ jq  '.[] | select (.dialogue_idx == "MUL1626.json")' dev_dials.json
    """
    multiwoz_2_1_dir = os.path.join(data_dir, "multiwoz_2_1")
    for dialogue_id in [
        "MUL1626.json",
        "PMUL3166.json",
        "MUL2258.json",
        "MUL2199.json",
        "MUL2096.json",
        "PMUL3470.json",
        "PMUL4478.json",
    ]:
        trade_dialogue_file = os.path.join(multiwoz_2_1_dir, dialogue_id)
        trade_dialogue = json.load(open(trade_dialogue_file))
        yield trade_dialogue


def test_create_programs_with_dummy_salience_model(data_dir: str):
    """Tests creating programs with a dummy salience model."""
    utterance_tokenizer = UtteranceTokenizer()
    salience_model = DummySalienceModel()
    expected_num_refer_calls = {
        "MUL1626.json": 0,
        "PMUL3166.json": 0,
        "MUL2258.json": 0,
        "MUL2199.json": 0,
        "MUL2096.json": 0,
        "PMUL3470.json": 0,
        "PMUL4478.json": 0,
    }
    for trade_dialogue in load_test_trade_dialogues(data_dir):
        dataflow_dialogue, num_refer_calls, _ = create_programs_for_trade_dialogue(
            trade_dialogue=trade_dialogue,
            keep_all_domains=True,
            remove_none=False,
            fill_none=False,
            salience_model=salience_model,
            no_revise=False,
            avoid_empty_plan=False,
            utterance_tokenizer=utterance_tokenizer,
        )
        dialogue_id = dataflow_dialogue.dialogue_id
        assert (
            num_refer_calls == expected_num_refer_calls[dialogue_id]
        ), "{} failed".format(dialogue_id)


def test_create_programs_without_revise(data_dir: str):
    """Tests creating programs without revise calls.

    It should not use refer calls even with a valid salience model.
    """
    utterance_tokenizer = UtteranceTokenizer()
    salience_model = VanillaSalienceModel()

    for trade_dialogue in load_test_trade_dialogues(data_dir):
        for avoid_empty_plan in [True, False]:
            _, num_refer_calls, _ = create_programs_for_trade_dialogue(
                trade_dialogue=trade_dialogue,
                keep_all_domains=True,
                remove_none=False,
                fill_none=False,
                salience_model=salience_model,
                no_revise=True,
                avoid_empty_plan=avoid_empty_plan,
                utterance_tokenizer=utterance_tokenizer,
            )
            assert num_refer_calls == 0


def test_create_programs_with_vanilla_salience_model(data_dir: str):
    """Tests creating programs with a vanilla salience model.
    """
    utterance_tokenizer = UtteranceTokenizer()
    salience_model = VanillaSalienceModel()
    expected_num_refer_calls = {
        "MUL1626.json": 1,
        "PMUL3166.json": 0,
        "MUL2258.json": 1,
        "MUL2199.json": 1,
        "MUL2096.json": 0,
        "PMUL3470.json": 0,
        "PMUL4478.json": 0,
    }
    for trade_dialogue in load_test_trade_dialogues(data_dir):
        dataflow_dialogue, num_refer_calls, _ = create_programs_for_trade_dialogue(
            trade_dialogue=trade_dialogue,
            keep_all_domains=True,
            remove_none=False,
            fill_none=False,
            salience_model=salience_model,
            no_revise=False,
            avoid_empty_plan=False,
            utterance_tokenizer=utterance_tokenizer,
        )
        dialogue_id = dataflow_dialogue.dialogue_id
        assert (
            num_refer_calls == expected_num_refer_calls[dialogue_id]
        ), "{} failed".format(dialogue_id)


def test_create_programs_with_revise(trade_dialogue_1: Dict[str, Any]):
    utterance_tokenizer = UtteranceTokenizer()
    salience_model = VanillaSalienceModel()
    expected_plans: List[str] = [
        # turn 1
        """(find (Constraint[Hotel] :name (?= "none") :type (?= "none")))""",
        #  turn 2
        """(ReviseConstraint :new (Constraint[Hotel] :name (?= "hilton") :pricerange (?= "cheap") :type (?= "guest house")) :oldLocation (Constraint[Constraint[Hotel]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 3
        """(ReviseConstraint :new (Constraint[Hotel] :name (?= "none")) :oldLocation (Constraint[Constraint[Hotel]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 4
        """(abandon (Constraint[Hotel]))""",
        # turn 5
        """(find (Constraint[Hotel] :area (?= "west")))""",
        # turn 6
        """(find (Constraint[Restaurant] :area (refer (Constraint[Area]))))""",
        # turn 7
        """(ReviseConstraint :new (Constraint[Restaurant] :pricerange (refer (Constraint[Pricerange]))) :oldLocation (Constraint[Constraint[Restaurant]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 8
        "()",
        # turn 9
        """(find (Constraint[Taxi] :departure (?= "none")))""",
        # turn 10
        "()",
    ]
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
    for turn, expected_lispress in zip(dataflow_dialogue.turns, expected_plans):
        lispress = turn.lispress
        assert lispress == expected_lispress


def test_create_programs_with_revise_with_fill_none(trade_dialogue_1: Dict[str, Any]):
    utterance_tokenizer = UtteranceTokenizer()
    salience_model = VanillaSalienceModel()

    expected_plans: List[str] = [
        # turn 1
        """(find (Constraint[Hotel] :area (?= "none") :book-day (?= "none") :book-people (?= "none") :book-stay (?= "none") :internet (?= "none") :name (?= "none") :parking (?= "none") :pricerange (?= "none") :stars (?= "none") :type (?= "none")))""",
        #  turn 2
        """(ReviseConstraint :new (Constraint[Hotel] :name (?= "hilton") :pricerange (?= "cheap") :type (?= "guest house")) :oldLocation (Constraint[Constraint[Hotel]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 3
        """(ReviseConstraint :new (Constraint[Hotel] :name (?= "none")) :oldLocation (Constraint[Constraint[Hotel]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 4
        """(abandon (Constraint[Hotel]))""",
        # turn 5
        """(find (Constraint[Hotel] :area (?= "west") :book-day (?= "none") :book-people (?= "none") :book-stay (?= "none") :internet (?= "none") :name (?= "none") :parking (?= "none") :pricerange (?= "none") :stars (?= "none") :type (?= "none")))""",
        # turn 6
        """(find (Constraint[Restaurant] :area (refer (Constraint[Area])) :book-day (?= "none") :book-people (?= "none") :book-time (?= "none") :food (?= "none") :name (?= "none") :pricerange (?= "none")))""",
        # turn 7
        """(ReviseConstraint :new (Constraint[Restaurant] :pricerange (refer (Constraint[Pricerange]))) :oldLocation (Constraint[Constraint[Restaurant]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 8
        "()",
        # turn 9
        """(find (Constraint[Taxi] :arriveby (?= "none") :departure (?= "none") :destination (?= "none") :leaveat (?= "none")))""",
        # turn 10
        "()",
    ]
    dataflow_dialogue, _, _ = create_programs_for_trade_dialogue(
        trade_dialogue=trade_dialogue_1,
        keep_all_domains=True,
        remove_none=False,
        fill_none=True,
        salience_model=salience_model,
        no_revise=False,
        avoid_empty_plan=False,
        utterance_tokenizer=utterance_tokenizer,
    )
    for turn, expected_plan in zip(
        dataflow_dialogue.turns, expected_plans  # pylint: disable=no-member
    ):
        lispress = turn.lispress
        assert lispress == expected_plan


def test_create_programs_with_revise_with_avoid_empty_plan(
    trade_dialogue_1: Dict[str, Any]
):
    utterance_tokenizer = UtteranceTokenizer()
    salience_model = VanillaSalienceModel()
    expected_plans: List[str] = [
        # turn 1
        """(find (Constraint[Hotel] :name (?= "none") :type (?= "none")))""",
        #  turn 2
        """(ReviseConstraint :new (Constraint[Hotel] :name (?= "hilton") :pricerange (?= "cheap") :type (?= "guest house")) :oldLocation (Constraint[Constraint[Hotel]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 3
        """(ReviseConstraint :new (Constraint[Hotel] :name (?= "none")) :oldLocation (Constraint[Constraint[Hotel]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 4
        """(abandon (Constraint[Hotel]))""",
        # turn 5
        """(find (Constraint[Hotel] :area (?= "west")))""",
        # turn 6
        """(find (Constraint[Restaurant] :area (refer (Constraint[Area]))))""",
        # turn 7
        """(ReviseConstraint :new (Constraint[Restaurant] :pricerange (refer (Constraint[Pricerange]))) :oldLocation (Constraint[Constraint[Restaurant]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 8
        """(ReviseConstraint :new (Constraint[Restaurant] :pricerange (refer (Constraint[Pricerange]))) :oldLocation (Constraint[Constraint[Restaurant]]) :rootLocation (roleConstraint #(Path "output")))""",
        # turn 9
        """(find (Constraint[Taxi] :departure (?= "none")))""",
        # turn 10
        """(ReviseConstraint :new (Constraint[Taxi] :departure (?= "none")) :oldLocation (Constraint[Constraint[Taxi]]) :rootLocation (roleConstraint #(Path "output")))""",
    ]
    dataflow_dialogue, _, _ = create_programs_for_trade_dialogue(
        trade_dialogue=trade_dialogue_1,
        keep_all_domains=True,
        remove_none=False,
        fill_none=False,
        salience_model=salience_model,
        no_revise=False,
        avoid_empty_plan=True,
        utterance_tokenizer=utterance_tokenizer,
    )
    for turn_part, expected_plan in zip(dataflow_dialogue.turns, expected_plans):
        lispress = turn_part.lispress
        assert lispress == expected_plan
