#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import json
import os

import jsons
from test_dataflow.multiwoz.test_create_programs import load_test_trade_dialogues

from dataflow.multiwoz.create_belief_state_prediction_report import (
    main as create_belief_state_prediction_report,
)
from dataflow.multiwoz.create_belief_state_tracker_data import (
    main as create_belief_state_tracker_data,
)
from dataflow.multiwoz.create_programs import main as create_programs
from dataflow.multiwoz.evaluate_belief_state_predictions import EvaluationStats
from dataflow.multiwoz.evaluate_belief_state_predictions import (
    main as evaluate_belief_state_predictions,
)
from dataflow.multiwoz.execute_programs import main as execute_programs
from dataflow.multiwoz.patch_trade_dialogues import main as patch_trade_dialogues


def test_cli_workflow(data_dir: str, tmp_path: str):
    """An end-to-end test on the CLI workflow.

    This test involves multiple CLI steps.
    1. patch_trade_dialogues
    2. create_programs
    3. execute_programs
    4. create_belief_state_tracker_data
    5. evaluate_belief_state_predictions

    It does not test all corner cases but it should catch some common errors in the workflow.
    """
    # ============
    # merges all test dialogues
    # ============
    trade_data_file = os.path.join(tmp_path, "s010.merged_trade_dials.jsonl")
    trade_dialogues = list(load_test_trade_dialogues(data_dir))
    with open(trade_data_file, "w") as fp:
        fp.write(json.dumps(trade_dialogues, indent=2))
        fp.write("\n")

    # ============
    # patches TRADE dialogues
    # ============
    patched_dials_file, _ = patch_trade_dialogues(
        trade_data_file=trade_data_file, outbase=os.path.join(tmp_path, "s020.merged")
    )

    # ============
    # create programs for TRADE dialogues
    # ============
    dataflow_dialogues = create_programs(
        trade_data_file=patched_dials_file,
        keep_all_domains=True,
        remove_none=False,
        fill_none=False,
        no_refer=False,
        no_revise=False,
        avoid_empty_plan=False,
        outbase=os.path.join(tmp_path, "s030.merged"),
    )

    # ============
    # execute programs
    # ============
    complete_execution_results_file, _, _ = execute_programs(
        dialogues_file=dataflow_dialogues,
        no_revise=False,
        no_refer=False,
        cheating_mode="never",
        cheating_execution_results_file=None,
        outbase=os.path.join(tmp_path, "s040.merged"),
    )
    another_complete_execution_results_file, _, _ = execute_programs(
        dialogues_file=dataflow_dialogues,
        no_revise=False,
        no_refer=False,
        cheating_mode="always",
        cheating_execution_results_file=complete_execution_results_file,
        outbase=os.path.join(tmp_path, "s040.merged"),
    )
    # because we use the execution results from the dataflow_dialogues itself as the cheating_execution_results_file,
    # the outcome should be identical
    for actual, expected in zip(
        open(another_complete_execution_results_file),
        open(complete_execution_results_file),
    ):
        assert actual == expected

    # ============
    # creates belief state tracker data
    # ============
    gold_data_file = os.path.join(tmp_path, "s050.merged_gold.data.jsonl")
    create_belief_state_tracker_data(
        trade_data_file=patched_dials_file,
        belief_state_tracker_data_file=gold_data_file,
    )
    prediction_report_jsonl = create_belief_state_prediction_report(
        input_data_file=complete_execution_results_file,
        file_format="dataflow",
        remove_none=False,
        gold_data_file=gold_data_file,
        outbase=os.path.join(tmp_path, "s050.merged_hypo"),
    )

    # ============
    # computes the accuracy
    # ============
    scores_file = evaluate_belief_state_predictions(
        prediction_report_jsonl=prediction_report_jsonl,
        outbase=os.path.join(tmp_path, "s060.merged"),
    )
    stats = jsons.loads(open(scores_file).read(), EvaluationStats)
    assert stats.accuracy == 1.0
    for _, accuracy in stats.accuracy_for_slot.items():
        assert accuracy == 1.0
