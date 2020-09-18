#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
#
#  From https://github.com/budzianowski/multiwoz:
#
#  The MIT License (MIT)
#
#  Copyright (c) 2019 Paweł Budzianowski
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# -*- coding: utf-8 -*-
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Download and pre-process MultiWOZ data.

This code is adapted from create_data.py in https://github.com/jasonwu0731/trade-dst (commit 8ace8c3),
which uses code from from https://github.com/budzianowski/multiwoz.
- Support both MultiWOZ 2.0 and 2.1.
- Cleaned up some unused code.
- Formatted the code to pass black & pylint checks.
- Add argparse.
"""

import argparse
import json
import os
import re
import shutil
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import requests
from tqdm import tqdm

np.set_printoptions(precision=3)

np.random.seed(2)


# GLOBAL VARIABLES
MAX_LENGTH = 50
IGNORE_KEYS_IN_GOAL = ["eod", "topic", "messageLen", "message"]

# url to the MultiWoZ 2.0 dataset
MULTIWOZ_2_0_URL = "https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y"
# url to the MultiWoZ 2.1 dataset
MULTIWOZ_2_1_URL = "https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y"


def _create_replacement_pair(line):
    tok_from, tok_to = line.replace("\n", "").split("\t")
    return " " + tok_from + " ", " " + tok_to + " "


REPLACEMENTS = [
    _create_replacement_pair(line)
    for line in open(os.path.join(os.path.dirname(__file__), "mapping.pair"), "r")
]


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if (
            sidx + 1 < len(text)
            and re.match("[0-9]", text[sidx - 1])
            and re.match("[0-9]", text[sidx + 1])
        ):
            sidx += 1
            continue
        if text[sidx - 1] != " ":
            text = text[:sidx] + " " + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != " ":
            text = text[: sidx + 1] + " " + text[sidx + 1 :]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r"^\s*|\s*$", "", text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # weird unicode bug
    text = re.sub("(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(";", ",")
    text = re.sub(r"$\/", "", text)
    text = text.replace("/", " and ")

    # replace other special characters
    text = text.replace("-", " ")
    text = re.sub(r'["\<>@\(\)]', "", text)  # remove

    # insert white space before and after tokens:
    for token in ["?", ".", ",", "!"]:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace("'s", text)

    # replace it's, does't, you'd ... etc
    text = re.sub(r"^'", "", text)
    text = re.sub(r"'$", "", text)
    text = re.sub(r"'\s", " ", text)
    text = re.sub(r"\s'", " ", text)
    for fromx, tox in REPLACEMENTS:
        text = " " + text + " "
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(" +", " ", text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(r"^\d+$", tokens[i]) and re.match(r"\d+$", tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = " ".join(tokens)

    return text


def fixDelex(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = data2[filename.strip(".json")][str(idx_acts)]
    except:
        return data

    if not isinstance(turn, str):  # and not isinstance(turn, unicode):
        for k, _act in turn.items():
            if "Attraction" in k:
                if "restaurant_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "restaurant", "attraction"
                    )
                if "hotel_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "hotel", "attraction"
                    )
            if "Hotel" in k:
                if "attraction_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "attraction", "hotel"
                    )
                if "restaurant_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "restaurant", "hotel"
                    )
            if "Restaurant" in k:
                if "attraction_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "attraction", "restaurant"
                    )
                if "hotel_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "hotel", "restaurant"
                    )

    return data


def getDialogueAct(
    filename, data, data2, idx, idx_acts
):  # pylint: disable=unused-argument
    """Given system dialogue acts fix automatic delexicalization."""
    acts = []
    try:
        turn = data2[filename.strip(".json")][str(idx_acts)]
    except:
        return acts

    if not isinstance(turn, str):  # and not isinstance(turn, unicode):
        for k in turn.keys():
            # temp = [k.split('-')[0].lower(), k.split('-')[1].lower()]
            # for a in turn[k]:
            #     acts.append(temp + [a[0].lower()])

            if k.split("-")[1].lower() == "request":
                for a in turn[k]:
                    acts.append(a[0].lower())
            elif k.split("-")[1].lower() == "inform":
                for a in turn[k]:
                    acts.append([a[0].lower(), normalize(a[1].lower())])

    return acts


def get_summary_bstate(bstate, get_domain=False):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [
        "taxi",
        "restaurant",
        "hospital",
        "hotel",
        "attraction",
        "train",
        "police",
    ]
    summary_bstate = []
    summary_bvalue = []
    active_domain = []
    for domain in domains:
        domain_active = False

        booking = []
        # print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]["book"].keys()):
            if slot == "booked":
                if len(bstate[domain]["book"]["booked"]) != 0:
                    booking.append(1)
                    # summary_bvalue.append("book {} {}:{}".format(domain, slot, "Yes"))
                else:
                    booking.append(0)
            else:
                if bstate[domain]["book"][slot] != "":
                    booking.append(1)
                    summary_bvalue.append(
                        [
                            "{}-book {}".format(domain, slot.strip().lower()),
                            normalize(bstate[domain]["book"][slot].strip().lower()),
                        ]
                    )  # (["book", domain, slot, bstate[domain]['book'][slot]])
                else:
                    booking.append(0)
        if domain == "train":
            if "people" not in bstate[domain]["book"].keys():
                booking.append(0)
            if "ticket" not in bstate[domain]["book"].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]["semi"]:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]["semi"][slot] == "not mentioned":
                slot_enc[0] = 1
            elif bstate[domain]["semi"][slot] in [
                "dont care",
                "dontcare",
                "don't care",
                "do not care",
            ]:
                slot_enc[1] = 1
                summary_bvalue.append(
                    ["{}-{}".format(domain, slot.strip().lower()), "dontcare"]
                )  # (["semi", domain, slot, "dontcare"])
            elif bstate[domain]["semi"][slot]:
                summary_bvalue.append(
                    [
                        "{}-{}".format(domain, slot.strip().lower()),
                        normalize(bstate[domain]["semi"][slot].strip().lower()),
                    ]
                )  # (["semi", domain, slot, bstate[domain]['semi'][slot]])
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
            active_domain.append(domain)
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    if get_domain:
        return active_domain
    else:
        return summary_bstate, summary_bvalue


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d["log"]) % 2 != 0:
        # print path
        print("odd # of turns")
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp["goal"] = d["goal"]  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    # last_bvs = []
    for i in range(len(d["log"])):
        if len(d["log"][i]["text"].split()) > maxlen:
            # print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = d["log"][i]["text"]
            if not is_ascii(text):
                # print('not ascii')
                return None
            usr_turns.append(d["log"][i])
        else:  # sys turn
            text = d["log"][i]["text"]
            if not is_ascii(text):
                # print('not ascii')
                return None
            belief_summary, belief_value_summary = get_summary_bstate(
                d["log"][i]["metadata"]
            )
            d["log"][i]["belief_summary"] = str(belief_summary)
            d["log"][i]["belief_value_summary"] = belief_value_summary
            sys_turns.append(d["log"][i])
    d_pp["usr_log"] = usr_turns
    d_pp["sys_log"] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t["text"] for t in d_orig["usr_log"]]
    sys = [t["text"] for t in d_orig["sys_log"]]
    sys_a = [t["dialogue_acts"] for t in d_orig["sys_log"]]
    bvs = [t["belief_value_summary"] for t in d_orig["sys_log"]]
    domain = [t["domain"] for t in d_orig["usr_log"]]
    for item in zip(usr, sys, sys_a, domain, bvs):
        dial.append(
            {
                "usr": item[0],
                "sys": item[1],
                "sys_a": item[2],
                "domain": item[3],
                "bvs": item[4],
            }
        )
    return dial


def download_data(url):
    """Downloads data from URL with a progress bar.
    See https://stackoverflow.com/a/37573701/11996682

    Returns the bytes.
    """
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="B", unit_scale=True)
    data = BytesIO()
    for block in r.iter_content(block_size):
        t.update(len(block))
        data.write(block)
    t.close()
    if total_size != 0 and t.n != total_size:  # pylint: disable=consider-using-in
        raise ValueError(f"Unexpected error during downloading {url}")
    return data


def loadData(output_dir, use_multiwoz_2_1=False):
    data_url = os.path.join(output_dir, "multi-woz", "data.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "multi-woz"))

    if not os.path.exists(data_url):
        print("Downloading and unzipping the MultiWOZ dataset")
        if not use_multiwoz_2_1:
            data = download_data(MULTIWOZ_2_0_URL)
            subdir_name = "MULTIWOZ2 2"
        else:
            data = download_data(MULTIWOZ_2_1_URL)
            subdir_name = "MULTIWOZ2.1"
        zip_ref = ZipFile(data)
        zip_ref.extractall(os.path.join(output_dir, "multi-woz"))
        zip_ref.close()
        # resp = urllib.request.urlopen(DATASET_URL)
        # data = resp.read()
        # zip_ref = ZipFile(BytesIO(data))
        # zip_ref.extractall(os.path.join(output_dir, "multi-woz"))
        # zip_ref.close()
        shutil.copy(
            os.path.join(output_dir, "multi-woz", subdir_name, "data.json"),
            os.path.join(output_dir, "multi-woz"),
        )
        shutil.copy(
            os.path.join(output_dir, "multi-woz", subdir_name, "valListFile.json"),
            os.path.join(output_dir, "multi-woz"),
        )
        shutil.copy(
            os.path.join(output_dir, "multi-woz", subdir_name, "testListFile.json"),
            os.path.join(output_dir, "multi-woz"),
        )
        shutil.copy(
            os.path.join(output_dir, "multi-woz", subdir_name, "dialogue_acts.json"),
            os.path.join(output_dir, "multi-woz"),
        )


def getDomain(idx, log, domains, last_domain):
    if idx == 1:
        active_domains = get_summary_bstate(log[idx]["metadata"], True)
        crnt_doms = active_domains[0] if len(active_domains) != 0 else domains[0]
        return crnt_doms
    else:
        ds_diff = get_ds_diff(log[idx - 2]["metadata"], log[idx]["metadata"])
        if len(ds_diff.keys()) == 0:  # no clues from dialog states
            crnt_doms = last_domain
        else:
            crnt_doms = list(ds_diff.keys())
        # print(crnt_doms)
        return crnt_doms[0]  # How about multiple domains in one sentence senario ?


def get_ds_diff(prev_d, crnt_d):
    diff = {}
    # Sometimes, metadata is an empty dictionary, bug?
    if not prev_d or not crnt_d:
        return diff

    for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
        assert k1 == k2
        if v1 != v2:  # updated
            diff[k2] = v2
    return diff


def createData(output_dir, use_multiwoz_2_1=False):
    # download the data
    loadData(output_dir, use_multiwoz_2_1)

    # create dictionary of delexicalied values that then we will search against, order matters here!
    # dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    fin1 = open(os.path.join(output_dir, "multi-woz", "data.json"), "r")
    data = json.load(fin1)

    fin2 = open(os.path.join(output_dir, "multi-woz", "dialogue_acts.json"), "r")
    data2 = json.load(fin2)

    print("Processing dialogues ...")
    for _didx, dialogue_name in tqdm(enumerate(data), unit=" dialogues"):

        dialogue = data[dialogue_name]

        domains = []
        for dom_k, dom_v in dialogue["goal"].items():
            if (
                dom_v and dom_k not in IGNORE_KEYS_IN_GOAL
            ):  # check whether contains some goal entities
                domains.append(dom_k)

        idx_acts = 1
        last_domain, _last_slot_fill = "", []
        for idx, turn in enumerate(dialogue["log"]):
            # normalization, split and delexicalization of the sentence
            origin_text = normalize(turn["text"])
            # origin_text = delexicalize.markEntity(origin_text, dic)
            dialogue["log"][idx]["text"] = origin_text

            if idx % 2 == 1:  # if it's a system turn

                cur_domain = getDomain(idx, dialogue["log"], domains, last_domain)
                last_domain = [cur_domain]

                dialogue["log"][idx - 1]["domain"] = cur_domain
                dialogue["log"][idx]["dialogue_acts"] = getDialogueAct(
                    dialogue_name, dialogue, data2, idx, idx_acts
                )
                idx_acts += 1

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)

        delex_data[dialogue_name] = dialogue

    return delex_data


def divideData(data, output_dir):
    """Given test and validation sets, divide
    the data for three different sets"""
    testListFile = []
    fin = open(os.path.join(output_dir, "multi-woz", "testListFile.json"), "r")
    for line in fin:
        testListFile.append(line[:-1])
    fin.close()

    valListFile = []
    fin = open(os.path.join(output_dir, "multi-woz", "valListFile.json"), "r")
    for line in fin:
        valListFile.append(line[:-1])
    fin.close()

    trainListFile = open(os.path.join(output_dir, "trainListFile"), "w")

    test_dials = []
    val_dials = []
    train_dials = []

    count_train, count_val, count_test = 0, 0, 0

    for dialogue_name in data:
        # print dialogue_name
        dial_item = data[dialogue_name]
        domains = []
        for dom_k, dom_v in dial_item["goal"].items():
            if (
                dom_v and dom_k not in IGNORE_KEYS_IN_GOAL
            ):  # check whether contains some goal entities
                domains.append(dom_k)

        dial = get_dial(data[dialogue_name])
        if dial:
            dialogue = {}
            dialogue["dialogue_idx"] = dialogue_name
            dialogue["domains"] = list(
                set(domains)
            )  # list(set([d['domain'] for d in dial]))
            last_bs = []
            dialogue["dialogue"] = []

            for turn_i, turn in enumerate(dial):
                # usr, usr_o, sys, sys_o, sys_a, domain
                turn_dialog = {}
                turn_dialog["system_transcript"] = (
                    dial[turn_i - 1]["sys"] if turn_i > 0 else ""
                )
                turn_dialog["turn_idx"] = turn_i
                turn_dialog["belief_state"] = [
                    {"slots": [s], "act": "inform"} for s in turn["bvs"]
                ]
                turn_dialog["turn_label"] = [
                    bs["slots"][0]
                    for bs in turn_dialog["belief_state"]
                    if bs not in last_bs
                ]
                turn_dialog["transcript"] = turn["usr"]
                turn_dialog["system_acts"] = (
                    dial[turn_i - 1]["sys_a"] if turn_i > 0 else []
                )
                turn_dialog["domain"] = turn["domain"]
                last_bs = turn_dialog["belief_state"]
                dialogue["dialogue"].append(turn_dialog)

            if dialogue_name in testListFile:
                test_dials.append(dialogue)
                count_test += 1
            elif dialogue_name in valListFile:
                val_dials.append(dialogue)
                count_val += 1
            else:
                trainListFile.write(dialogue_name + "\n")
                train_dials.append(dialogue)
                count_train += 1

    print(
        "# of dialogues: Train {}, Val {}, Test {}".format(
            count_train, count_val, count_test
        )
    )

    # save all dialogues
    with open(os.path.join(output_dir, "dev_dials.json"), "w") as f:
        json.dump(val_dials, f, indent=4)

    with open(os.path.join(output_dir, "test_dials.json"), "w") as f:
        json.dump(test_dials, f, indent=4)

    with open(os.path.join(output_dir, "train_dials.json"), "w") as f:
        json.dump(train_dials, f, indent=4)


def main(output_dir, use_multiwoz_2_1):
    print("Create WOZ-like dialogues. Get yourself a coffee, this might take a while.")
    delex_data = createData(output_dir, use_multiwoz_2_1)
    print("Divide dialogues ...")
    divideData(delex_data, output_dir)


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    cmdline_parser.add_argument(
        "--use_multiwoz_2_1",
        action="store_true",
        default=False,
        help="If True, use MultiWOZ 2.1 instead of MultiWoZ 2.0.",
    )
    cmdline_parser.add_argument("--output_dir", help="output directory")
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(args.output_dir, args.use_multiwoz_2_1)
