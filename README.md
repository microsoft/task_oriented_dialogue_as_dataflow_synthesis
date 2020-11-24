# Task-Oriented Dialogue as Dataflow Synthesis
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="https://avatars2.githubusercontent.com/u/9585815?s=200&v=4" width="18%">


This repository contains tools and instructions for reproducing the experiments in the paper
**Task-Oriented Dialogue as Dataflow Synthesis** (TACL 2020).
If you use any source code or data included in this toolkit in your work, please cite the following paper.
```bib
@article{SMDataflow2020,
  author = {{Semantic Machines} and Andreas, Jacob and Bufe, John and Burkett, David and Chen, Charles and Clausman, Josh and Crawford, Jean and Crim, Kate and DeLoach, Jordan and Dorner, Leah and Eisner, Jason and Fang, Hao and Guo, Alan and Hall, David and Hayes, Kristin and Hill, Kellie and Ho, Diana and Iwaszuk, Wendy and Jha, Smriti and Klein, Dan and Krishnamurthy, Jayant and Lanman, Theo and Liang, Percy and Lin, Christopher H. and Lintsbakh, Ilya and McGovern, Andy and Nisnevich, Aleksandr and Pauls, Adam and Petters, Dmitrij and Read, Brent and Roth, Dan and Roy, Subhro and Rusak, Jesse and Short, Beth and Slomin, Div and Snyder, Ben and Striplin, Stephon and Su, Yu and Tellman, Zachary and Thomson, Sam and Vorobev, Andrei and Witoszko, Izabela and Wolfe, Jason and Wray, Abby and Zhang, Yuchen and Zotov, Alexander},
  title = {Task-Oriented Dialogue as Dataflow Synthesis},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {8},
  pages = {556--571},
  year = {2020},
  month = sep,
  url = {https://doi.org/10.1162/tacl_a_00333},
  abstract = {We describe an approach to task-oriented dialogue in which dialogue state is represented as a dataflow graph. A dialogue agent maps each user utterance to a program that extends this graph. Programs include metacomputation operators for reference and revision that reuse dataflow fragments from previous turns. Our graph-based state enables the expression and manipulation of complex user intents, and explicit metacomputation makes these intents easier for learned models to predict. We introduce a new dataset, SMCalFlow, featuring complex dialogues about events, weather, places, and people. Experiments show that dataflow graphs and metacomputation substantially improve representability and predictability in these natural dialogues. Additional experiments on the MultiWOZ dataset show that our dataflow representation enables an otherwise off-the-shelf sequence-to-sequence model to match the best existing task-specific state tracking model. The SMCalFlow dataset, code for replicating experiments, and a public leaderboard are available at \url{https://www.microsoft.com/en-us/research/project/dataflow-based-dialogue-semantic-machines}.},
}
```

## Install
```bash
# (Recommended) Create a virtual environment
virtualenv --python=python3 env
source env/bin/activate

# Install the sm-dataflow package and its core dependencies
pip install git+https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis.git

# Download the spaCy model for tokenization
python -m spacy download en_core_web_md-2.2.0 --direct

# Install OpenNMT-py and PyTorch for training and running the models
pip install OpenNMT-py==1.0.0 torch==1.4.0
```
* Our experiments used OpenNMT-py 1.0.0 with PyTorch 1.4.0. Other versions are not tested. 
You can skip these two packages if you don't need to train or run the models.

## SMCalFlow Experiments
Follow the steps below to reproduce the results reported in the paper (Table 2).

1. Download the SMCalFlow dataset on [this page](https://microsoft.github.io/task_oriented_dialogue_as_dataflow_synthesis/).

2. Compute data statistics:
    ```bash
    dataflow_dialogues_stats_dir="output/dataflow_dialogues_stats"
    mkdir -p "${dataflow_dialogues_stats_dir}"
    python -m dataflow.analysis.compute_data_statistics \
        --dataflow_dialogues_dir ${dataflow_dialogues_dir} \
        --subset train valid \
        --outdir ${dataflow_dialogues_stats_dir}
    ```
    * Basic statistics
    
        |           | num_dialogues  | num_turns | num_kept_turns | num_skipped_turns | num_refer_turns | num_revise_turns |
        | --------- | :-:            | :-:       | :-:            | :-:               | :-:             | :-:              |
        | **train** | 32,647         | 133,821   | 121,200        | 12,621            | 33,011          | 9,315            |
        | **valid** | 3,649          | 14,757    | 13,499         | 1,258             | 3,544           | 1,052            |
        | **test**  | 5,211          | 22,012    | 21,224         | 7,88              | 8,965           | 3,315            |
        | **all**   | 41,517         | 170,590   | 155,923        | 14,667            | 45,520          | 13,682           |
        
        * We currently do not release the test set, but we report the data statistics here.
        * **NOTE**: There are a small number of turns (`num_skipped_turns` in the table) whose sole purpose is to establish dialogue context and should not be directly trained or tested on. The dataset statistics reported in the paper are based on non-skipped turns only. 
        
3. Prepare text data for the OpenNMT toolkit.
    ```bash
    onmt_text_data_dir="output/onmt_text_data"
    mkdir -p "${onmt_text_data_dir}"
    for subset in "train" "valid"; do
        python -m dataflow.onmt_helpers.create_onmt_text_data \
            --dialogues_jsonl ${dataflow_dialogues_dir}/${subset}.dataflow_dialogues.jsonl \
            --num_context_turns 2 \
            --include_program \
            --include_described_entities \
            --onmt_text_data_outbase ${onmt_text_data_dir}/${subset}
    done
    ```
    * We use `--include_program` to add the gold program of the context turns.
    * We use `--include_described_entities` to add the entities (e.g., `entity@123456`) described in the generation 
    outcome for the context turns. These entities mentioned in the context turns can appear in the "inlined" programs
    for the current turn, and thus, we include them in the source sequence so that the seq2seq model can produce such
    tokens via a copy mechanism.
    * You can vary the number of context turns by changing `--num_context_turns`.

4. Compute statistics for the created OpenNMT text data.
    ```bash
    onmt_data_stats_dir="output/onmt_data_stats"
    mkdir -p "${onmt_data_stats_dir}"
    python -m dataflow.onmt_helpers.compute_onmt_data_stats \
        --text_data_dir ${onmt_text_data_dir} \
        --suffix src src_tok tgt \
        --subset train valid \
        --outdir ${onmt_data_stats_dir}
    ```

5. Train OpenNMT models. You can also skip this step and instead download the trained model from the table below.
    ```bash
    onmt_binarized_data_dir="output/onmt_binarized_data"
    mkdir -p "${onmt_binarized_data_dir}"
   
    src_tok_max_ntokens=$(jq '."100"' ${onmt_data_stats_dir}/train.src_tok.ntokens_stats.json)
    tgt_max_ntokens=$(jq '."100"' ${onmt_data_stats_dir}/train.tgt.ntokens_stats.json)
   
    # create OpenNMT binarized data
    onmt_preprocess \
        --dynamic_dict \
        --train_src ${onmt_text_data_dir}/train.src_tok \
        --train_tgt ${onmt_text_data_dir}/train.tgt \
        --valid_src ${onmt_text_data_dir}/valid.src_tok \
        --valid_tgt ${onmt_text_data_dir}/valid.tgt \
        --src_seq_length ${src_tok_max_ntokens} \
        --tgt_seq_length ${tgt_max_ntokens} \
        --src_words_min_frequency 0 \
        --tgt_words_min_frequency 0 \
        --save_data ${onmt_binarized_data_dir}/data

    # extract pretrained Glove 840B embeddings (https://nlp.stanford.edu/projects/glove/)
    glove_840b_dir="output/glove_840b"
    mkdir -p "${glove_840b_dir}"
    wget -O ${glove_840b_dir}/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip ${glove_840b_dir}/glove.840B.300d.zip -d ${glove_840b_dir}

    onmt_embeddings_dir="output/onmt_embeddings"
    mkdir -p "${onmt_embeddings_dir}"
    python -m dataflow.onmt_helpers.embeddings_to_torch \
        -emb_file_both ${glove_840b_dir}/glove.840B.300d.txt \
        -dict_file ${onmt_binarized_data_dir}/data.vocab.pt \
        -output_file ${onmt_embeddings_dir}/embeddings

    # train OpenNMT models
    onmt_models_dir="output/onmt_models"
    mkdir -p "${onmt_models_dir}"
   
    batch_size=64
    train_num_datapoints=$(jq '.train' ${onmt_data_stats_dir}/nexamples.json)
    # validate approximately at each epoch
    valid_steps=$(python3 -c "from math import ceil; print(ceil(${train_num_datapoints}/${batch_size}))")
   
    onmt_train \
        --encoder_type brnn \
        --decoder_type rnn \
        --rnn_type LSTM \
        --global_attention general \
        --global_attention_function softmax \
        --generator_function softmax \
        --copy_attn_type general \
        --copy_attn \
        --seed 1 \
        --optim adam \
        --learning_rate 0.001 \
        --early_stopping 2 \
        --batch_size ${batch_size} \
        --valid_batch_size 8 \
        --valid_steps ${valid_steps} \
        --save_checkpoint_steps ${valid_steps} \
        --data ${onmt_binarized_data_dir}/data \
        --pre_word_vecs_enc ${onmt_embeddings_dir}/embeddings.enc.pt \
        --pre_word_vecs_dec ${onmt_embeddings_dir}/embeddings.dec.pt \
        --word_vec_size 300 \
        --attention_dropout 0 \
        --dropout 0.5 \
        --layers ??? \
        --rnn_size ??? \
        --gpu_ranks 0 \
        --world_size 1 \
        --save_model ${onmt_models_dir}/checkpoint 
    ```
    * Hyperparameters for models reported in the Table 2 in the paper.
    
        |           | `--layers` | `--rnn_size` | model |
        | --------- | :-:        | :-:          | :-:   |
        | dataflow  |     2      |     384      | [link](https://smresearchstorage.blob.core.windows.net/smcalflow-public/smcalflow.full.checkpoint_last.pt) |
        | inline    |     3      |     384      | [link](https://smresearchstorage.blob.core.windows.net/smcalflow-public/smcalflow.inlined.checkpoint_last.pt) |

6. Make predictions using a trained OpenNMT model. You need to replace the `checkpoint_last.pt` in the following script 
with the final model you get from the previous step.
    ```bash
    onmt_translate_outdir="output/onmt_translate_output"
    mkdir -p "${onmt_translate_outdir}"
   
    onmt_model_pt="${onmt_models_dir}/checkpoint_last.pt"
    nbest=5
    tgt_max_ntokens=$(jq '."100"' ${onmt_data_stats_dir}/train.tgt.ntokens_stats.json)
   
    # predict programs using a trained OpenNMT model
    onmt_translate \
        --model ${onmt_model_pt} \
        --max_length ${tgt_max_ntokens} \
        --src ${onmt_text_data_dir}/valid.src_tok \
        --replace_unk \
        --n_best ${nbest} \
        --batch_size 8 \
        --beam_size 10 \
        --gpu 0 \
        --report_time \
        --output ${onmt_translate_outdir}/valid.nbest
    ```
   
7. Compute the exact-match accuracy (taking into account whether the `program_execution_oracle.refer_are_correct` is `true`).
    ```bash
    evaluation_outdir="output/evaluation_output"
    mkdir -p "${evaluation_outdir}"
   
    # create the prediction report
    python -m dataflow.onmt_helpers.create_onmt_prediction_report \
        --dialogues_jsonl ${dataflow_dialogues_dir}/valid.dataflow_dialogues.jsonl \
        --datum_id_jsonl ${onmt_text_data_dir}/valid.datum_id \
        --src_txt ${onmt_text_data_dir}/valid.src_tok \
        --ref_txt ${onmt_text_data_dir}/valid.tgt \
        --nbest_txt ${onmt_translate_outdir}/valid.nbest \
        --nbest ${nbest} \
        --outbase ${evaluation_outdir}/valid
   
    # evaluate the predictions (all turns)
    python -m dataflow.onmt_helpers.evaluate_onmt_predictions \
        --prediction_report_tsv ${evaluation_outdir}/valid.prediction_report.tsv \
        --scores_json ${evaluation_outdir}/valid.all.scores.json
   
    # evaluate the predictions (refer turns)
    python -m dataflow.onmt_helpers.evaluate_onmt_predictions \
        --prediction_report_tsv ${evaluation_outdir}/valid.prediction_report.tsv \
        --datum_ids_json ${dataflow_dialogues_stats_dir}/valid.refer_turn_ids.jsonl \
        --scores_json ${evaluation_outdir}/valid.refer_turns.scores.json
   
    # evaluate the predictions (revise turns)
    python -m dataflow.onmt_helpers.evaluate_onmt_predictions \
        --prediction_report_tsv ${evaluation_outdir}/valid.prediction_report.tsv \
        --datum_ids_json ${dataflow_dialogues_stats_dir}/valid.revise_turn_ids.jsonl \
        --scores_json ${evaluation_outdir}/valid.revise_turns.scores.json
    ```
   
8. Calculate the statistical significance for two different experiments.
    ```bash
    analysis_outdir="output/analysis_output"
    mkdir -p "${analysis_outdir}"
    python -m dataflow.analysis.calculate_statistical_significance \
        --exp0_prediction_report_tsv ${exp0_evaluation_outdir}/valid.prediction_report.tsv \
        --exp1_prediction_report_tsv ${exp1_evaluation_outdir}/valid.prediction_report.tsv \
        --scores_json ${analysis_outdir}/exp0_vs_exp1.valid.scores.json
    ```
    * The `exp0_evaluation_outdir` and `exp1_evaluation_outdir` are the `evaluation_outdir` in Step 7 for corresponding experiments. 
    * You can also provide `--datum_ids_jsonl` to carry out the significance test on a subset of turns.

## MultiWOZ Experiments
1. Download the MultiWoZ dataset and convert it to dataflow programs.
    ```bash
    # creates TRADE-processed dialogues
    raw_trade_dialogues_dir="output/trade_dialogues"
    mkdir -p "${raw_trade_dialogues_dir}"
    python -m dataflow.multiwoz.trade_dst.create_data \
        --use_multiwoz_2_1 \
        --output_dir ${raw_trade_dialogues_dir}

    # patch TRADE dialogues
    patched_trade_dialogues_dir="output/patched_trade_dialogues"
    mkdir -p "${patched_trade_dialogues_dir}"
    for subset in "train" "dev" "test"; do
        python -m dataflow.multiwoz.patch_trade_dialogues \
            --trade_data_file ${raw_trade_dialogues_dir}/${subset}_dials.json \
            --outbase ${patched_trade_dialogues_dir}/${subset}
    done
    ln -sr ${patched_trade_dialogues_dir}/dev_dials.json ${patched_trade_dialogues_dir}/valid_dials.json

    # create dataflow programs
    dataflow_dialogues_dir="output/dataflow_dialogues"
    mkdir -p "${dataflow_dialogues_dir}"
    for subset in "train" "valid" "test"; do
        python -m dataflow.multiwoz.create_programs \
            --trade_data_file ${patched_trade_dialogues_dir}/${subset}_dials.json \
            --outbase ${dataflow_dialogues_dir}/${subset}
    done
    ```
    * To create programs that inline `refer` calls, add `--no_refer` when running the `dataflow.multiwoz.create_programs` command. 
    * To create programs that inline both `refer` and `revise` calls, add `--no_refer --no_revise`.

2. Prepare text data for the OpenNMT toolkit.
    ```bash
    onmt_text_data_dir="output/onmt_text_data"
    mkdir -p "${onmt_text_data_dir}"
    for subset in "train" "valid" "test"; do
        python -m dataflow.onmt_helpers.create_onmt_text_data \
            --dialogues_jsonl ${dataflow_dialogues_dir}/${subset}.dataflow_dialogues.jsonl \
            --num_context_turns 2 \
            --include_agent_utterance \
            --onmt_text_data_outbase ${onmt_text_data_dir}/${subset}
    done
    ```
    * We use `--include_agent_utterance` following the setup in [TRADE](https://github.com/jasonwu0731/trade-dst) (Wu et al., 2019).
    * You can vary the number of context turns by changing `--num_context_turns`.

3. Compute statistics for the created OpenNMT text data.
    ```bash
    onmt_data_stats_dir="output/onmt_data_stats"
    mkdir -p "${onmt_data_stats_dir}"
    python -m dataflow.onmt_helpers.compute_onmt_data_stats \
        --text_data_dir ${onmt_text_data_dir} \
        --suffix src src_tok tgt \
        --subset train valid test \
        --outdir ${onmt_data_stats_dir}
    ```

4. Train OpenNMT models. You can also skip this step and instead download the trained models from the table below.
    ```bash
    onmt_binarized_data_dir="output/onmt_binarized_data"
    mkdir -p "${onmt_binarized_data_dir}"
   
    # create OpenNMT binarized data
    src_tok_max_ntokens=$(jq '."100"' ${onmt_data_stats_dir}/train.src_tok.ntokens_stats.json)
    tgt_max_ntokens=$(jq '."100"' ${onmt_data_stats_dir}/train.tgt.ntokens_stats.json)
   
    onmt_preprocess \
        --dynamic_dict \
        --train_src ${onmt_text_data_dir}/train.src_tok \
        --train_tgt ${onmt_text_data_dir}/train.tgt \
        --valid_src ${onmt_text_data_dir}/valid.src_tok \
        --valid_tgt ${onmt_text_data_dir}/valid.tgt \
        --src_seq_length ${src_tok_max_ntokens} \
        --tgt_seq_length ${tgt_max_ntokens} \
        --src_words_min_frequency 0 \
        --tgt_words_min_frequency 0 \
        --save_data ${onmt_binarized_data_dir}/data

    # extract pretrained Glove 6B embeddings
    glove_6b_dir="output/glove_6b"
    mkdir -p "${glove_6b_dir}"
    wget -O ${glove_6b_dir}/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
    unzip ${glove_6b_dir}/glove.6B.zip -d ${glove_6b_dir}

    onmt_embeddings_dir="output/onmt_embeddings"
    mkdir -p "${onmt_embeddings_dir}"
    python -m dataflow.onmt_helpers.embeddings_to_torch \
        -emb_file_both ${glove_6b_dir}/glove.6B.300d.txt \
        -dict_file ${onmt_binarized_data_dir}/data.vocab.pt \
        -output_file ${onmt_embeddings_dir}/embeddings

    # train OpenNMT models
    onmt_models_dir="output/onmt_models"
    mkdir -p "${onmt_models_dir}"
   
    batch_size=64
    train_num_datapoints=$(jq '.train' ${onmt_data_stats_dir}/nexamples.json)
    # approximately validate at each epoch
    valid_steps=$(python3 -c "from math import ceil; print(ceil(${train_num_datapoints}/${batch_size}))")
   
    onmt_train \
        --encoder_type brnn \
        --decoder_type rnn \
        --rnn_type LSTM \
        --global_attention general \
        --global_attention_function softmax \
        --generator_function softmax \
        --copy_attn_type general \
        --copy_attn \
        --seed 1 \
        --optim adam \
        --learning_rate 0.001 \
        --early_stopping 2 \
        --batch_size ${batch_size} \
        --valid_batch_size 8 \
        --valid_steps ${valid_steps} \
        --save_checkpoint_steps ${valid_steps} \
        --data ${onmt_binarized_data_dir}/data \
        --pre_word_vecs_enc ${onmt_embeddings_dir}/embeddings.enc.pt \
        --pre_word_vecs_dec ${onmt_embeddings_dir}/embeddings.dec.pt \
        --word_vec_size 300 \
        --attention_dropout 0 \
        --dropout ??? \
        --layers ??? \
        --rnn_size ??? \
        --gpu_ranks 0 \
        --world_size 1 \
        --save_model ${onmt_models_dir}/checkpoint 
    ```
    * Hyperparameters for models reported in the Table 3 in the paper.
    
        |                 | `--dropout` | `--layers` | `--rnn_size` | model |
        | ---------       | :-:        | :-:          | :-:         | :-:   |
        | dataflow (`--num_context_turns 2`)     | 0.7 | 2 | 384 | [link](https://smresearchstorage.blob.core.windows.net/smcalflow-public/multiwoz.full.checkpoint_last.pt) |
        | inline refer (`--num_context_turns 4`) | 0.3 | 3 | 320 | [link](https://smresearchstorage.blob.core.windows.net/smcalflow-public/multiwoz.inline_refer.checkpoint_last.pt) |
        | inline both (`--num_context_turns 10`) | 0.7 | 2 | 320 | [link](https://smresearchstorage.blob.core.windows.net/smcalflow-public/multiwoz.inline_both.checkpoint_last.pt) |

5. Make predictions using a trained OpenNMT model. You need to replace the `checkpoint_last.pt` in the following script with the actual model you get from
 the previous step.
    ```bash
    onmt_translate_outdir="output/onmt_translate_output"
    mkdir -p "${onmt_translate_outdir}"
   
    onmt_model_pt="${onmt_models_dir}/checkpoint_last.pt"
    nbest=5
    tgt_max_ntokens=$(jq '."100"' ${onmt_data_stats_dir}/train.tgt.ntokens_stats.json)
   
    # predict programs on the test set using a trained OpenNMT model
    onmt_translate \
        --model ${onmt_model_pt} \
        --max_length ${tgt_max_ntokens} \
        --src ${onmt_text_data_dir}/test.src_tok \
        --replace_unk \
        --n_best ${nbest} \
        --batch_size 8 \
        --beam_size 10 \
        --gpu 0 \
        --report_time \
        --output ${onmt_translate_outdir}/test.nbest
    ```

6. Compute the exact-match accuracy of the program predictions.
    ```
    evaluation_outdir="output/evaluation_output"
    mkdir -p "${evaluation_outdir}"
   
    # create the prediction report
    python -m dataflow.onmt_helpers.create_onmt_prediction_report \
        --dialogues_jsonl ${dataflow_dialogues_dir}/test.dataflow_dialogues.jsonl \
        --datum_id_jsonl ${onmt_text_data_dir}/test.datum_id \
        --src_txt ${onmt_text_data_dir}/test.src_tok \
        --ref_txt ${onmt_text_data_dir}/test.tgt \
        --nbest_txt ${onmt_translate_outdir}/test.nbest \
        --nbest ${nbest} \
        --outbase ${evaluation_outdir}/test
   
    # evaluate the predictions
    python -m dataflow.onmt_helpers.evaluate_onmt_predictions \
        --prediction_report_tsv ${evaluation_outdir}/test.prediction_report.tsv \
        --scores_json ${evaluation_outdir}/test.scores.json
    ```

7. Evaluate the belief state predictions. 
    ```bash
    belief_state_tracker_eval_dir="output/belief_state_tracker_eval"
    mkdir -p "${belief_state_tracker_eval_dir}"
   
    # creates the gold file from TRADE-preprocessed dialogues (after patch)
    python -m dataflow.multiwoz.create_belief_state_tracker_data \
        --trade_data_file ${patched_trade_dialogues_dir}/test_dials.json \
        --belief_state_tracker_data_file ${belief_state_tracker_eval_dir}/test.belief_state_tracker_data.jsonl
   
    # creates the hypo file from predicted programs
    python -m dataflow.multiwoz.execute_programs \
        --dialogues_file ${evaluation_outdir}/test.dataflow_dialogues.jsonl \
        --cheating_mode never \
        --outbase ${belief_state_tracker_eval_dir}/test.hypo
   
    python -m dataflow.multiwoz.create_belief_state_prediction_report \
        --input_data_file ${belief_state_tracker_eval_dir}/test.hypo.execution_results.jsonl \
        --format dataflow \
        --remove_none \
        --gold_data_file ${belief_state_tracker_eval_dir}/test.belief_state_tracker_data.jsonl \
        --outbase ${belief_state_tracker_eval_dir}/test
   
    # evaluates belief state predictions
    python -m dataflow.multiwoz.evaluate_belief_state_predictions \
        --prediction_report_jsonl ${belief_state_tracker_eval_dir}/test.prediction_report.jsonl \
        --outbase ${belief_state_tracker_eval_dir}/test
    ```
    * The scores are reported in `${belief_state_tracker_eval_dir}/test.scores.json`.
7. Calculate the statistical significance for two different experiments.
    ```bash
    analysis_outdir="output/analysis_output"
    mkdir -p "${analysis_outdir}"
    python -m dataflow.analysis.calculate_statistical_significance \
        --exp0_prediction_report_tsv ${exp0_evaluation_outdir}/test.prediction_report.tsv \
        --exp1_prediction_report_tsv ${exp1_evaluation_outdir}/test.prediction_report.tsv \
        --scores_json ${analysis_outdir}/exp0_vs_exp1.test.scores.json
    ```
    * The `exp0_evaluation_outdir` and `exp1_evaluation_outdir` are the `belief_state_tracker_eval_dir` in Step 7 for corresponding experiments. 

## Understand SMCalFlow Programs

Please read [this document](./README-LISPRESS.md) to understand the
syntax of SMCalFlow programs, and read [this document](./README-SEMANTICS.md)
to understand their semantics.