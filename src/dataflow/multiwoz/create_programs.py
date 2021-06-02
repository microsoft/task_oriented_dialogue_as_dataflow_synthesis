#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Converts TRADE-processed MultiWoZ data to programs that build dataflow graphs.

Here, we use best-effort conversion which ensures that the round-trip is always correct.
With a better salience model, we should see an increase in the number of refer calls in the data.
"""
import argparse
import json
import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, TextIO, Tuple

import jsons
from tqdm import tqdm

from dataflow.core.dialogue import Dialogue, ProgramExecutionOracle, Turn
from dataflow.core.lispress import program_to_lispress, render_compact
from dataflow.core.program import CallLikeOp, Expression, Program
from dataflow.core.program_utils import (
    DataflowFn,
    mk_call_op,
    mk_constraint,
    mk_equality_constraint,
    mk_revise_the_main_constraint,
    mk_salience,
    mk_unset_constraint,
    mk_value_op,
)
from dataflow.core.utterance_tokenizer import UtteranceTokenizer
from dataflow.core.utterance_utils import build_agent_utterance, build_user_utterance
from dataflow.multiwoz.execute_programs import (
    execute_program_for_turn,
    update_execution_trace,
)
from dataflow.multiwoz.ontology import DATAFLOW_SLOT_NAMES_FOR_DOMAIN
from dataflow.multiwoz.salience_model import (
    DummySalienceModel,
    ExecutionTrace,
    SalienceModelBase,
    VanillaSalienceModel,
)
from dataflow.multiwoz.trade_dst_utils import (
    concatenate_system_and_user_transcript,
    flatten_belief_state,
    get_domain_and_slot_name,
)


def mentioned_in_text(value: str, text: str) -> bool:
    """Returns true if the value is mentioned in text.

    Use some heuristics to increase the hit rate.
    """
    if re.search(r"\b{}\b".format(re.escape(value)), text):
        return True
    if value == "1":
        for word in ["single", "one"]:
            if re.search(r"\b{}\b".format(re.escape(word)), text):
                return True
    if re.match(r"0(\d):(\d)+", value):
        # e.g, "09:15" -> "9:15"
        if re.search(r"\b{}\b".format(re.escape(value[1:])), text):
            return True
    if value == "guest house":
        if re.search(r"\bguesthouse\b", text):
            return True
    return False


def logic_diff(
    old_belief_dict: Dict[str, str],
    new_belief_dict: Dict[str, str],
    transcript: Optional[str],
) -> List[Dict[str, str]]:
    """Computes the logic difference between the old belief state and the new belief state.

    TODO: This method should be refactored when we have time.
     - Should change diff_dicts to be keyed by diff-type.
      { $diffType: [ (topic, slot_fullname, slot_value) ] }
     - Should merge logic_diff with aggregate_logic_diff_by_topic.
     - Should use a Enum to define all the diff types.
    """
    diff_dicts: List[Dict[str, str]] = []
    slot_fullnames = set(old_belief_dict.keys()) | set(new_belief_dict.keys())

    # sort the slot_fullnames so the results are ordered
    for slot_fullname in sorted(slot_fullnames):
        domain = get_domain_and_slot_name(slot_fullname)[0]
        if slot_fullname not in old_belief_dict and slot_fullname in new_belief_dict:
            diff_dicts.append(
                {
                    "difftype": "added",
                    "topic": domain,
                    slot_fullname: new_belief_dict[slot_fullname],
                }
            )
        elif slot_fullname not in new_belief_dict and slot_fullname in old_belief_dict:
            diff_dicts.append(
                {
                    "difftype": "deleted",
                    "topic": domain,
                    slot_fullname: old_belief_dict[slot_fullname],
                }
            )
        else:
            old_slot_value = old_belief_dict[slot_fullname]
            new_slot_value = new_belief_dict[slot_fullname]
            if old_slot_value != new_slot_value:
                diff_dicts.append(
                    {
                        "difftype": "changed",
                        "topic": domain,
                        slot_fullname: new_slot_value,
                    }
                )
            elif (
                transcript is not None
                and new_slot_value != "yes"
                and mentioned_in_text(value=new_slot_value, text=transcript)
            ):
                # NOTE: We include the re-mentioned value in the logic difference.
                diff_dicts.append(
                    {
                        "difftype": "repeated",
                        "topic": domain,
                        slot_fullname: new_slot_value,
                    }
                )
    return diff_dicts


def aggregate_by_topic(
    diff_dict_list: List[Dict[str, str]]
) -> DefaultDict[str, Dict[str, Optional[str]]]:
    """Aggregates logic differences by topic."""
    diff_dict_for_topic: DefaultDict[str, Dict[str, Optional[str]]] = defaultdict(dict)
    for diff_dict in diff_dict_list:
        deletion = bool(diff_dict["difftype"] == "deleted")
        for k, v in diff_dict.items():
            if k not in ["difftype", "topic"]:
                if not deletion:
                    diff_dict_for_topic[diff_dict["topic"]][k] = v
                else:
                    diff_dict_for_topic[diff_dict["topic"]][k] = None
    return diff_dict_for_topic


def compute_logic_diff_by_topic(
    curr_belief_dict: Dict[str, str],
    last_belief_dict: Dict[str, str],
    trade_turn: Dict[str, Any],
    avoid_empty_plan: bool,
    last_nonempty_diff_dicts: List[Dict[str, str]],
    topic_keys: Dict[str, Set[str]],
) -> Tuple[DefaultDict[str, Dict[str, Optional[str]]], List[Dict[str, str]]]:
    """Computes the logic difference between the flatten belief states between the current turn and previous turn.

    Args:
        curr_belief_dict:
        last_belief_dict:
        trade_turn:
        avoid_empty_plan:
        last_nonempty_diff_dicts:
        topic_keys: (updated)
    Returns:
        A dictionary (topic, logic difference for the topic).
    """
    # If last_belief_dict == curr_belief_dict (and curr_belief_dict is not empty),
    # we set the difftype to 'unchanged' so we can use a revise call instead of creating an empty plan.
    if avoid_empty_plan and curr_belief_dict and last_belief_dict == curr_belief_dict:
        diff_dicts: List[Dict[str, str]] = []
        for diff_dict in last_nonempty_diff_dicts:
            # makes a copy of the diff dict so we do not change the last_nonempty_diff_dicts
            diff_dict = dict(diff_dict)
            # change "added" to "repeated" so we use "ReviseConstraint" instead of "Find"
            if diff_dict["difftype"] == "added":
                diff_dict["difftype"] = "repeated"
            diff_dicts.append(diff_dict)
    else:
        diff_dicts = logic_diff(
            old_belief_dict=last_belief_dict,
            new_belief_dict=curr_belief_dict,
            transcript=trade_turn["transcript"],
        )

    for diff_dict in diff_dicts:
        if diff_dict["difftype"] == "deleted":
            if diff_dict["topic"] not in topic_keys:
                # the domain may have been deleted if we use `avoid_empty_plan`
                continue
            for k, _ in diff_dict.items():
                if k not in ["difftype", "topic"]:
                    if k in topic_keys[diff_dict["topic"]]:
                        # the value may have been deleted if we use `avoid_empty_plan`
                        topic_keys[diff_dict["topic"]].remove(k)

    for diff_dict in diff_dicts:
        if diff_dict["difftype"] in ["repeated", "changed"]:
            for k, _ in diff_dict.items():
                if k not in ["difftype", "topic"]:
                    assert k in topic_keys[diff_dict["topic"]]

    for diff_dict in diff_dicts:
        if diff_dict["difftype"] == "added":
            for k, _ in diff_dict.items():
                if k not in ["difftype", "topic"]:
                    topic_keys[diff_dict["topic"]].add(k)

    return aggregate_by_topic(diff_dicts), diff_dicts


def generate_express_for_topic(
    topic: str,
    kvs: Dict[str, Optional[str]],
    text: str,
    execution_trace: ExecutionTrace,
    salience_model: SalienceModelBase,
    latest_pointer: Optional[int],
    pointer_count: int,
    no_revise: bool,
    is_abandon: bool,
) -> Tuple[List[Expression], int, List[Dict[str, str]]]:
    """Generates express for given topic and key value pairs kvs.

    Args:
        topic: Topic of express
        latest_pointer: Latest pointer to node of current topic
        kvs: Dictionary of key value pairs
        execution_trace: the ExecutionTrace at the previous turn, used for refer calls
        text: Text in scope used to search for the value
        pointer_count: Current pointer count, used to generate unique index
        no_revise: If True, do not use revise calls
        salience_model: The salience model to be used for refer calls
    Returns:
        A tuple of (a list of SerializedExpression, current pointer count, failed refer calls).
    """
    expressions: List[Expression] = []

    # a map from slot name to the pointer count (expressionId) of the slot value (including None) in the dataflow graph
    pointer_count_for_slot: Dict[str, int] = {}
    failed_refer_calls: List[Dict[str, str]] = []

    if not is_abandon:
        for slot_fullname, slot_value in sorted(kvs.items()):
            _domain, slot_name = get_domain_and_slot_name(slot_fullname)
            if slot_value is None:
                # None means the slot is deleted
                expression, pointer_count = mk_unset_constraint(idx=pointer_count)
                expressions.append(expression)
                pointer_count_for_slot[slot_name] = pointer_count
                continue

            slot_value = slot_value.lower()
            assert slot_value != ""

            # Best-effort conversion, i.e., use refer call only if
            # 1. the value is not mentioned in the text (current turn);
            # 2. revise call is allowed;
            # 3. the salience model can return the right value.
            use_refer = False
            if not mentioned_in_text(value=slot_value, text=text) and not no_revise:
                salience_value = salience_model.get_salient_value(
                    target_type=slot_name,
                    execution_trace=execution_trace,
                    exclude_values=set(),
                )
                if salience_value == slot_value:
                    use_refer = True
                else:
                    # records the failed salience calls so we can improve the salience model
                    failed_refer_calls.append(
                        {
                            "topic": topic,
                            "slotName": slot_name,
                            "targetSalienceValue": slot_value,
                            "returnedSalienceValue": salience_value,
                        }
                    )

            if use_refer:
                refer_expressions, pointer_count = mk_salience(
                    tpe=slot_name, idx=pointer_count
                )
                expressions.extend(refer_expressions)
            else:
                expression, pointer_count = mk_value_op(
                    value=slot_value, schema="String", idx=pointer_count,
                )
                expressions.append(expression)
                expression, pointer_count = mk_equality_constraint(
                    val=pointer_count, idx=pointer_count
                )
                expressions.append(expression)

            pointer_count_for_slot[slot_name] = pointer_count

    expression, pointer_count = mk_constraint(
        tpe=topic, args=list(pointer_count_for_slot.items()), idx=pointer_count
    )
    expressions.append(expression)

    if is_abandon:
        expression, pointer_count = mk_call_op(
            name=DataflowFn.Abandon.value, args=[pointer_count], idx=pointer_count
        )
        expressions.append(expression)

    elif latest_pointer is None or no_revise:
        expression, pointer_count = mk_call_op(
            name=DataflowFn.Find.value, args=[pointer_count], idx=pointer_count
        )
        expressions.append(expression)

    else:
        revise_expressions, pointer_count = mk_revise_the_main_constraint(
            tpe=topic, new_idx=pointer_count
        )
        expressions.extend(revise_expressions)

    return expressions, pointer_count, failed_refer_calls


def is_refer_call(expression: Expression) -> bool:
    return (
        isinstance(expression.op, CallLikeOp)
        and expression.op.name == DataflowFn.Refer.value
    )


def create_program_for_trade_turn(
    trade_turn: Dict[str, Any],
    last_belief_dict: Dict[str, str],
    curr_belief_dict: Dict[str, str],
    topic_pointers: Dict[str, Any],
    topic_keys: Dict[str, Set[str]],
    pointer_count: int,
    execution_trace: ExecutionTrace,
    salience_model: SalienceModelBase,
    no_revise: bool,
    avoid_empty_plan: bool,
    last_nonempty_diff_dicts: List[Dict[str, str]],
) -> Tuple[List[Expression], int, Dict[str, Any], List[Dict[str, str]]]:
    """Creates programs for a TRADE turn.

    Args:
        trade_turn: the current TRADE turn to be converted
        last_belief_dict: the flatten (and label-corrected) belief state at the previous turn
        curr_belief_dict: the flatten (and label-corrected) belief state at the current turn
        topic_pointers: the {domain: expressionId} map representing the active domains in the dataflow graph
        topic_keys: the {domain: { slot_name: slot_values } } map, records all active slot values
        pointer_count: the expressionId offset
        execution_trace: the execution trace of the dialogue at the end the previous turn
        salience_model: the salience model used for the conversion
        no_revise: If True, do not use revise call in the dataflow representation
        avoid_empty_plan: True if we do not generate empty plan
        last_nonempty_diff_dicts: last non-empty belief state diff

    Returns:
        A tuple of (
         a list of Expressions describing the program,
         the expressionId offset after the current turn,
         the refer call report,
         the nonempty belief state diff dict
        )
    """
    if no_revise:
        last_belief_dict = {}

    diff_dict_for_topic, last_nonempty_diff_dicts = compute_logic_diff_by_topic(
        curr_belief_dict=curr_belief_dict,
        last_belief_dict=last_belief_dict,
        trade_turn=trade_turn,
        avoid_empty_plan=avoid_empty_plan,
        last_nonempty_diff_dicts=last_nonempty_diff_dicts,
        topic_keys=topic_keys,
    )

    expressions: List[Expression] = []
    failed_refer_calls_for_topic: Dict[str, List[Dict[str, str]]] = {}
    text: str = concatenate_system_and_user_transcript(trade_turn)
    for topic, diff_dict in sorted(diff_dict_for_topic.items()):
        # generate the expressions for each topic one by one
        if not diff_dict:
            continue

        if topic not in topic_pointers or no_revise:
            topic_pointer = None
        else:
            topic_pointer = topic_pointers[topic]

        # We use abandon() when all slots of a domain get deleted
        num_topic_slots_in_last_belief = len(
            [k for k, v in last_belief_dict.items() if k.startswith(topic)]
        )
        is_abandon = (
            len(diff_dict) > 0
            and all([v is None for k, v in diff_dict.items()])
            and len(diff_dict) == num_topic_slots_in_last_belief
        )

        (
            expressions_for_topic,
            pointer_count,
            failed_refer_calls,
        ) = generate_express_for_topic(
            topic=topic,
            latest_pointer=topic_pointer,
            kvs=diff_dict,
            execution_trace=execution_trace,
            salience_model=salience_model,
            text=text,
            pointer_count=pointer_count,
            no_revise=no_revise,
            is_abandon=is_abandon,
        )
        topic_pointers[topic] = pointer_count

        expressions.extend(expressions_for_topic)
        if failed_refer_calls:
            failed_refer_calls_for_topic[topic] = failed_refer_calls

    # Removes dead topics from the topic_pointers so that next time the domain is activated, we should use
    # "Find" instead of "Revise".
    dead_topics: List[str] = [
        topic
        for topic, _pointer in topic_pointers.items()
        if not topic_keys.get(topic, set())
    ]
    for topic in dead_topics:
        # remove the topic from topic_pointers so new slot values will be added via "Find" instead of "Revise"
        del topic_pointers[topic]

    # Creates the refer call report for analyzing failed refer calls.
    # We will notice that a lot of refer call attempts are false alarms because our `mentioned_in_text` method
    # is very primitive. For example, `multi sports` will not match `multiple sports`.
    # A better logging in future would be identifying truely failed refer calls.
    num_refer_calls = 0
    for expression in expressions:
        if is_refer_call(expression):
            num_refer_calls += 1

    refer_call_report = {
        "turnIndex": trade_turn["turn_idx"],
        "prevAgentUtterance": trade_turn["system_transcript"],
        "currUserUtterance": trade_turn["transcript"],
        "lastBeliefDict": last_belief_dict,
        "beliefDictDiff": diff_dict_for_topic,
        "numResolveCalls": num_refer_calls,
        "failedResolveCalls": failed_refer_calls_for_topic,
    }
    return expressions, pointer_count, refer_call_report, last_nonempty_diff_dicts


def create_programs_for_trade_dialogue(
    trade_dialogue: Dict[str, Any],
    keep_all_domains: bool,
    remove_none: bool,
    fill_none: bool,
    salience_model: SalienceModelBase,
    no_revise: bool,
    avoid_empty_plan: bool,
    utterance_tokenizer: UtteranceTokenizer,
) -> Tuple[Dialogue, int, List[Dict[str, Any]]]:
    """Creates programs for a TRADE dialogue.

    Returns:
        A tuple of (Dialogue, the number of refer calls, refer call report at each turn).
    """
    if remove_none:
        assert not fill_none

    # the execution trace of the program for the dialogue
    # updated at the end of each trade_turn
    last_execution_trace = ExecutionTrace(slot_values=dict())
    # the number of refer calls in the program
    num_refer_calls: int = 0
    # the flatten belief state at the previous turn
    last_belief_dict: Dict[str, str] = {}

    pointer_count: int = 0
    topic_pointers: Dict[str, Any] = {}
    topic_keys: DefaultDict[str, Set[str]] = defaultdict(set)

    dataflow_turns: List[Turn] = []
    refer_call_reports: List[Dict[str, Any]] = []
    last_nonempty_diff_dicts: List[Dict[str, str]] = []

    trade_turns = trade_dialogue["dialogue"]
    assert not trade_turns[0][
        "system_transcript"
    ].strip(), "the leading agent utterance should be empty"

    for turn_index, trade_turn in enumerate(trade_turns):
        curr_belief_dict = flatten_belief_state(
            belief_state=trade_turn["belief_state"],
            keep_all_domains=keep_all_domains,
            remove_none=remove_none,
        )
        if fill_none:
            # Sometimes the user may activate a domain without any constraint,
            # in this case, all slot values may be "none" (or "dontcare"?).
            active_domains: Set[str] = {
                get_domain_and_slot_name(slot_fullname=slot_fullname)[0]
                for slot_fullname, slot_value in curr_belief_dict.items()
            }
            for domain in active_domains:
                # adds "none" to activate domains so the model can get more supervision signals
                # note slots in inactivate domains are not added
                for slot_name in DATAFLOW_SLOT_NAMES_FOR_DOMAIN[domain]:
                    slot_fullname = "{}-{}".format(domain, slot_name)
                    if slot_fullname not in curr_belief_dict:
                        curr_belief_dict[slot_fullname] = "none"

        (
            expressions,
            pointer_count,
            refer_call_report,
            last_nonempty_diff_dicts,
        ) = create_program_for_trade_turn(
            trade_turn=trade_turn,
            curr_belief_dict=curr_belief_dict,
            last_belief_dict=last_belief_dict,
            topic_pointers=topic_pointers,
            topic_keys=topic_keys,
            execution_trace=last_execution_trace,
            pointer_count=pointer_count,
            salience_model=salience_model,
            no_revise=no_revise,
            avoid_empty_plan=avoid_empty_plan,
            last_nonempty_diff_dicts=last_nonempty_diff_dicts,
        )

        program = Program(expressions=expressions)
        lispress = render_compact(program_to_lispress(program))

        dataflow_turn = Turn(
            turn_index=trade_turn["turn_idx"],
            user_utterance=build_user_utterance(
                text=trade_turn["transcript"], utterance_tokenizer=utterance_tokenizer
            ),
            # NOTE: The agentUtterance should be the one following the user trade_turn.
            # In the original MultiWoZ data, there is an ending agent utterance. This agent utterance is
            # removed in the TRADE preprocessing script because it doesn't change the belief state.
            # For now, we use a special NULL string for the last trade_turn.
            agent_utterance=build_agent_utterance(
                text=trade_turns[turn_index + 1]["system_transcript"]
                if turn_index + 1 < len(trade_turns)
                else "",
                utterance_tokenizer=utterance_tokenizer,
                described_entities=[],
            ),
            lispress=lispress,
            skip=False,
            program_execution_oracle=ProgramExecutionOracle(
                # no exception for MultiWoZ
                has_exception=False,
                # all refer calls are correct since we use a best-effort conversion
                refer_are_correct=True,
            ),
        )
        dataflow_turns.append(dataflow_turn)

        refer_call_report["dialogueId"] = trade_dialogue["dialogue_idx"]
        refer_call_reports.append(refer_call_report)
        num_refer_calls += refer_call_report["numResolveCalls"]

        # update the execution trace
        reconstructed_belief_dict, execution_result = execute_program_for_turn(
            curr_turn=dataflow_turn,
            last_execution_trace=last_execution_trace,
            last_belief_dict=None if no_revise else last_belief_dict,
            salience_model=salience_model,
        )
        # makes sure the round-trip is successful
        assert (
            reconstructed_belief_dict == curr_belief_dict
        ), "turn {} in {} does not round-trip".format(
            trade_turn["turn_idx"], trade_dialogue["dialogue_idx"]
        )

        last_execution_trace = update_execution_trace(
            last_execution_trace=last_execution_trace,
            partial_execution_result=execution_result,
        )

        last_belief_dict = curr_belief_dict

    dataflow_dialogue = Dialogue(
        dialogue_id=trade_dialogue["dialogue_idx"], turns=dataflow_turns
    )
    return dataflow_dialogue, num_refer_calls, refer_call_reports


def save_refer_call_report_txt(
    refer_call_report: Dict[str, Any], txt_fp: TextIO
) -> None:
    if not refer_call_report.get("failedResolveCalls"):
        return

    txt_fp.write("======================\n")
    for key in ["dialogueId", "turnIndex", "prevAgentUtterance", "currUserUtterance"]:
        txt_fp.write("{}\t{}\n".format(key, refer_call_report.get(key)))
    txt_fp.write("=== lastBeliefDict ===\n")
    for key, val in sorted(refer_call_report.get("lastBeliefDict").items()):
        txt_fp.write("{}={}\n".format(key, val))
    txt_fp.write("=== beliefDictDiff ===\n")
    for topic, diff in sorted(refer_call_report.get("beliefDictDiff").items()):
        for key, val in sorted(diff.items()):
            txt_fp.write("{}={}\n".format(key, val))
    txt_fp.write("=== failedResolveCalls ===\n")
    for topic, failed_refer_calls in sorted(
        refer_call_report.get("failedResolveCalls").items()
    ):
        txt_fp.write("TOPIC: {}\n".format(topic))
        for failed_refer_call in failed_refer_calls:
            txt_fp.write(json.dumps(failed_refer_call, indent=2))
            txt_fp.write("\n")


def main(
    trade_data_file: str,
    keep_all_domains: bool,
    remove_none: bool,
    fill_none: bool,
    no_refer: bool,
    no_revise: bool,
    avoid_empty_plan: bool,
    outbase: str,
) -> str:
    utterance_tokenizer = UtteranceTokenizer()

    salience_model: SalienceModelBase
    if no_refer:
        salience_model = DummySalienceModel()
    else:
        salience_model = VanillaSalienceModel()

    total_num_refer_calls = 0
    num_dialogues = 0
    num_dialogues_without_refer_calls = 0
    num_turns_with_refer: int = 0
    num_turns: int = 0

    trade_dialogues = json.load(open(trade_data_file, "r"))
    dataflow_dialogues_file = outbase + ".dataflow_dialogues.jsonl"
    dataflow_dialogues_jsonl_fp = open(dataflow_dialogues_file, "w")
    refer_call_reports_jsonl_fp = open(outbase + ".refer_call_reports.jsonl", "w")
    refer_call_reports_txt_fp = open(outbase + ".refer_call_reports.txt", "w")

    for trade_dialogue in tqdm(trade_dialogues, unit=" dialogues"):
        (
            dataflow_dialogue,
            num_refer_calls,
            refer_call_reports,
        ) = create_programs_for_trade_dialogue(
            trade_dialogue=trade_dialogue,
            keep_all_domains=keep_all_domains,
            remove_none=remove_none,
            fill_none=fill_none,
            salience_model=salience_model,
            no_revise=no_revise,
            avoid_empty_plan=avoid_empty_plan,
            utterance_tokenizer=utterance_tokenizer,
        )
        dataflow_dialogues_jsonl_fp.write(jsons.dumps(dataflow_dialogue) + "\n")
        for refer_call_report in refer_call_reports:
            if refer_call_report["numResolveCalls"] > 0:
                num_turns_with_refer += 1
            num_turns += 1

            refer_call_reports_jsonl_fp.write(json.dumps(refer_call_report))
            refer_call_reports_jsonl_fp.write("\n")
            save_refer_call_report_txt(refer_call_report, refer_call_reports_txt_fp)

        total_num_refer_calls += num_refer_calls
        num_dialogues += 1
        if num_refer_calls == 0:
            num_dialogues_without_refer_calls += 1
    dataflow_dialogues_jsonl_fp.close()
    refer_call_reports_jsonl_fp.close()
    refer_call_reports_txt_fp.close()

    # print some basic statistics
    print(f"Converted {num_dialogues} dialogues")
    print(
        f"Number of dialogues without refer calls: {num_dialogues_without_refer_calls}"
    )
    print(f"Number of turns: {num_turns}")
    print(f"Number of turns with refer calls: {num_turns_with_refer}")
    print(f"Total number of refer calls: {total_num_refer_calls}")

    return dataflow_dialogues_file


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--trade_data_file", help="the trade data file to be converted"
    )
    argument_parser.add_argument(
        "--keep_all_domains",
        default=False,
        action="store_true",
        help="if True, keep all domains; otherwise only keep the 5 domains used in TRADE",
    )
    argument_parser.add_argument(
        "--remove_none",
        default=False,
        action="store_true",
        help='if True, remove slots with "none" value from the belief state',
    )
    argument_parser.add_argument(
        "--no_refer",
        default=False,
        action="store_true",
        help="if True, do not use refer calls",
    )
    argument_parser.add_argument(
        "--no_revise",
        default=False,
        action="store_true",
        help="if True, do not use revise calls",
    )
    argument_parser.add_argument(
        "--avoid_empty_plan",
        default=False,
        action="store_true",
        help="if True, avoid creating empty plan",
    )
    argument_parser.add_argument(
        "--fill_none",
        default=False,
        action="store_true",
        help='if True, fill "none" to unspecified slots in active domains',
    )
    argument_parser.add_argument("--outbase", help="output files basename")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(argument_parser=cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        trade_data_file=args.trade_data_file,
        keep_all_domains=args.keep_all_domains,
        remove_none=args.remove_none,
        fill_none=args.fill_none,
        no_refer=args.no_refer,
        no_revise=args.no_revise,
        avoid_empty_plan=args.avoid_empty_plan,
        outbase=args.outbase,
    )
