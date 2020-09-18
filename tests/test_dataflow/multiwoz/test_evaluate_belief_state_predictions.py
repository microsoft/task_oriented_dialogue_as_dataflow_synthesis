#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from dataflow.multiwoz.evaluate_belief_state_predictions import EvaluationStats


def test_evaluation_stats():
    stats = EvaluationStats()
    assert stats.accuracy == 0
    assert not stats.accuracy_for_slot

    another_stats = EvaluationStats(
        num_total_turns=5,
        num_correct_turns=2,
        num_correct_turns_for_slot={"a": 1, "b": 2},
        num_total_dialogues=2,
        num_correct_dialogues=1,
    )
    stats += another_stats
    assert stats == another_stats
    assert stats.accuracy == 0.4
    assert set(stats.accuracy_for_slot.keys()) == {"a", "b"}
    assert stats.accuracy_for_slot["a"] == 0.2
    assert stats.accuracy_for_slot["b"] == 0.4
