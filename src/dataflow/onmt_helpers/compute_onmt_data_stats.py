#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Computes statistics on the data created by create_onmt_text_data.
"""
import argparse
import json
import os
import typing
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd


def compute_num_examples(input_txt: str) -> int:
    """Computes number of examples in a file.

    Simply count lines since each line is an example.
    """
    count = 0
    for _ in open(input_txt):
        count += 1
    return count


def compute_ntokens_percentiles(
    input_txt_files: List[str], percentiles: List[int]
) -> List[int]:
    """Computes the percentiles of sequence lengths."""
    ntokens_array: List[int] = sum(
        [
            [len(line.strip().split()) for line in open(input_txt)]
            for input_txt in input_txt_files
        ],
        [],
    )
    return np.percentile(ntokens_array, percentiles).tolist()


def save_ntokens_percentiles(
    percentiles: List[int], values: List[int], stats_tsv: str, stats_json: str
) -> None:
    """Saves the sequence length percentiles to a tsv file."""
    stats_df = pd.DataFrame({"percentile": percentiles, "value": values})
    stats_df.to_csv(stats_tsv, sep="\t", index=False, encoding="utf-8")
    stats_dict = dict(zip(percentiles, values))
    with open(stats_json, "w") as fp:
        fp.write(json.dumps(stats_dict, indent=2))
        fp.write("\n")


def compute_token_occurrences(
    input_txt_files: List[str],
) -> typing.DefaultDict[str, int]:
    """Computes token occurrences."""
    token_counter: typing.DefaultDict[str, int] = defaultdict(int)
    for input_txt in input_txt_files:
        for line in open(input_txt):
            for token in line.strip().split():
                token_counter[token] += 1
    return token_counter


def save_token_occurrences(
    token_counter: typing.DefaultDict[str, int], counts_tsv: str
) -> None:
    counts_df = pd.DataFrame(
        [
            {"token": token, "count": count}
            for token, count in sorted(token_counter.items(), key=lambda x: -x[1])
        ]
    )
    counts_df.to_csv(counts_tsv, sep="\t", index=False, encoding="utf-8")


def main(
    text_data_dir: str, subsets: List[str], suffixes: List[str], outdir: str,
) -> None:
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # ===============
    # nexamples stats
    # ===============
    nexamples_lookup = {
        subset: compute_num_examples(
            os.path.join(text_data_dir, "{}.datum_id".format(subset))
        )
        for subset in subsets
    }
    with open(os.path.join(outdir, "nexamples.json"), "w") as fp:
        json.dump(nexamples_lookup, fp, indent=2)

    # ===============
    # ntokens stats
    # ===============
    percentiles = list(range(0, 101, 10))
    for suffix in suffixes:
        for subset in subsets:
            input_txt = os.path.join(text_data_dir, f"{subset}.{suffix}")
            outbase = os.path.join(outdir, f"{subset}.{suffix}")

            values = compute_ntokens_percentiles(
                input_txt_files=[input_txt], percentiles=percentiles
            )
            save_ntokens_percentiles(
                percentiles=percentiles,
                values=values,
                stats_tsv=f"{outbase}.ntokens_stats.tsv",
                stats_json=f"{outbase}.ntokens_stats.json",
            )
            counter = compute_token_occurrences(input_txt_files=[input_txt])
            save_token_occurrences(
                token_counter=counter, counts_tsv=f"{outbase}.token_count.tsv"
            )

        if len(subsets) > 1:
            input_txt_files = [
                os.path.join(text_data_dir, f"{subset}.{suffix}") for subset in subsets
            ]
            outbase = os.path.join(outdir, "{}.{}".format("-".join(subsets), suffix))

            values = compute_ntokens_percentiles(
                input_txt_files=input_txt_files, percentiles=percentiles
            )
            save_ntokens_percentiles(
                percentiles=percentiles,
                values=values,
                stats_tsv=f"{outbase}.ntokens_stats.tsv",
                stats_json=f"{outbase}.ntokens_stats.json",
            )

            counter = compute_token_occurrences(input_txt_files=input_txt_files)
            save_token_occurrences(
                token_counter=counter, counts_tsv=f"{outbase}.token_count.tsv"
            )


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument("--text_data_dir", help="the text data directory")
    argument_parser.add_argument(
        "--suffix",
        nargs="+",
        default=["src", "src_tok", "tgt"],
        help="the suffix to be analyzed",
    )
    argument_parser.add_argument(
        "--subset", nargs="+", help="the subset to be analyzed"
    )
    argument_parser.add_argument("--outdir", help="the output directory")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        text_data_dir=args.text_data_dir,
        subsets=args.subset,
        suffixes=args.suffix,
        outdir=args.outdir,
    )
