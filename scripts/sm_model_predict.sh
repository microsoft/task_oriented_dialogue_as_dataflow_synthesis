#!/bin/bash
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
#
#  Semantic Machines (TM) software.
#
#  This script takes as input a model path and validation set, and generates predictions on the validation set in
#  a file named `predictions.jsonl`.

model_path=$1
data_path=$2

pip install --user .
pip install --user OpenNMT-py==1.0.0
export PATH=$PATH:.local/bin

# Prepare text data for the OpenNMT toolkit.
onmt_text_data_dir="output/onmt_text_data"
mkdir -p "${onmt_text_data_dir}"
subset="valid"
python -m dataflow.leaderboard.create_text_data \
    --dialogues_jsonl ${data_path} \
    --num_context_turns 2 \
    --include_program \
    --include_described_entities \
    --onmt_text_data_outbase ${onmt_text_data_dir}/${subset}

# Make predictions using a trained OpenNMT model. You need to replace the `checkpoint_last.pt` in the following script
#with the final model you get from the previous step.
onmt_translate_outdir="output/onmt_translate_output"
mkdir -p "${onmt_translate_outdir}"

nbest=5
tgt_max_ntokens=1000

# predict programs using a trained OpenNMT model
onmt_translate \
    --model ${model_path} \
    --max_length ${tgt_max_ntokens} \
    --src ${onmt_text_data_dir}/valid.src_tok \
    --replace_unk \
    --n_best ${nbest} \
    --batch_size 8 \
    --beam_size 10 \
    --gpu 0 \
    --report_time \
    --output ${onmt_translate_outdir}/valid.nbest

# create the prediction report
python -m dataflow.leaderboard.predict \
    --datum_id_jsonl ${onmt_text_data_dir}/valid.datum_id \
    --src_txt ${onmt_text_data_dir}/valid.src_tok \
    --ref_txt ${onmt_text_data_dir}/valid.tgt \
    --nbest_txt ${onmt_translate_outdir}/valid.nbest \
    --nbest ${nbest}

rm -rf output .local .cache
