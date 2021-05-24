#!/bin/bash
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
#
#  Semantic Machines (TM) software.
#
#  This script converts the SMCalFlow native format to the format used by the leaderboard.
#  Only runs for valid and test.

data_folder=$1
output_folder=$2

for subset in "valid" "test"; do
    python -m dataflow.leaderboard.create_leaderboard_data \
        --dialogues_jsonl ${data_folder}/${subset}.dataflow_dialogues.jsonl \
        --contextualized_turns_file ${output_folder}/${subset}.leaderboard_dialogues.jsonl \
        --turn_answers_file ${output_folder}/${subset}.answers.jsonl  \
        --dialogue_id_prefix ${subset}
done
