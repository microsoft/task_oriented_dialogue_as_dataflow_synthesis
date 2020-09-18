#!/bin/bash
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
#
#  Semantic Machines (TM) software.
#
#  This script takes as input a gold answers file and a prediction file, and outputs the accuracy.

gold_file=$1
prediction_file=$2

pip install --user .
export PATH=$PATH:.local/bin

python -m dataflow.leaderboard.evaluate  --predictions_jsonl ${prediction_file} --gold_jsonl ${gold_file} --scores_json scores.json

rm -rf output .local .cache
