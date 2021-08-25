#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Semantic Machines\N{TRADE MARK SIGN} software.

Executes programs to produce TRADE belief states.
"""
import argparse
import copy
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jsons
import numpy as np

from dataflow.core.dialogue import Dialogue, Turn
from dataflow.core.io_utils import load_jsonl_file, load_jsonl_file_and_build_lookup
from dataflow.core.program import BuildStructOp, CallLikeOp, Expression, ValueOp
from dataflow.core.program_utils import DataflowFn
from dataflow.multiwoz.salience_model import (
    DummySalienceModel,
    ExecutionTrace,
    PartialExecutionResult,
    SalienceModelBase,
    VanillaSalienceModel,
)


@dataclass(frozen=True)
class CompleteExecutionResult:
    dialogue_id: str
    turn_index: int
    belief_dict: Dict[str, str]
    execution_trace: ExecutionTrace
    # records the partial execution result for debugging purpose
    partial_execution_result: PartialExecutionResult


def get_constraint_schema(op_schema: str) -> str:
    if op_schema.startswith("Constraint["):
        if op_schema.startswith("Constraint[Constraint["):
            schema = op_schema[len("Constraint[Constraint[") : -1]
        else:
            schema = op_schema[len("Constraint[") : -1]
        return schema.replace("_", "-")
    raise ValueError(f"Unknown constraint schema: {op_schema}")


def execute_expression(
    expression: Expression,
    salience_model: SalienceModelBase,
    last_execution_trace: ExecutionTrace,
    last_partial_execution_result: PartialExecutionResult,
) -> PartialExecutionResult:
    """Executes the expression and updates the partial execution result.

    We assume the expressions are executed in an order such that all values/constraints needed by the current expression
    are already evaluated and saved in the partial_execution_result. Thus, we do not need to recursively evaluate the
    arguments of a STRUCT_OP.

    Args:
        expression: the current expression to be executed
        last_execution_trace: the context execution trace up to the preceding turn
        salience_model: the salience model used by the executor
        last_partial_execution_result: the partial execution result up to the last expression in the current turn
    Returns:
        the new partial execution result after executing the current expression
    """
    op = expression.op
    partial_execution_result = copy.deepcopy(last_partial_execution_result)

    if isinstance(op, ValueOp):
        value = json.loads(op.value)
        underlying = value.get("underlying")
        schema = value.get("schema")
        if schema != "Unit":
            assert schema in ["String", "Path"], "invalid value: {}".format(value)
            partial_execution_result.values[expression.id] = underlying

    elif isinstance(op, BuildStructOp):
        if op.op_schema == "EmptyConstraint":
            partial_execution_result.values[expression.id] = None
        elif op.op_schema.startswith("Constraint["):
            domain = get_constraint_schema(op.op_schema).lower()
            partial_execution_result.constraints[expression.id] = domain

            for field, arg_id in zip(op.op_fields, expression.arg_ids):
                if field not in partial_execution_result.slot_values:
                    partial_execution_result.slot_values[field] = []
                partial_execution_result.slot_values[field].append(
                    (partial_execution_result.values[arg_id], arg_id)
                )
    elif isinstance(op, CallLikeOp):
        if op.name == DataflowFn.Abandon.value:
            partial_execution_result.constraints[
                expression.id
            ] = partial_execution_result.constraints[expression.arg_ids[0]]
        if op.name == "?=":
            arg_id = expression.arg_ids[0]
            partial_execution_result.values[
                expression.id
            ] = partial_execution_result.values[arg_id]
        if op.name == DataflowFn.Refer.value:
            arg_id = expression.arg_ids[0]
            if arg_id in partial_execution_result.constraints:
                target_type = partial_execution_result.constraints[arg_id]
                salience_value = salience_model.get_salient_value(
                    target_type=target_type,
                    execution_trace=last_execution_trace,
                    exclude_values=set(),
                )
                partial_execution_result.values[expression.id] = salience_value
                partial_execution_result.refer_calls[expression.id] = target_type
                # we do not need to update partial_execution_result.slot_values here because it will
                # be added in the Constraint call.
    else:
        raise ValueError("Unexpected expression: {}".format(expression))

    return partial_execution_result


def update_belief_state(
    curr_turn: Turn,
    curr_values: Dict[str, str],
    curr_constraint_types: Dict[str, str],
    last_belief_dict: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Returns the belief state for the current turn part."""
    curr_belief_dict: Dict[str, str] = {}
    if last_belief_dict is not None:
        curr_belief_dict.update(last_belief_dict)

    for expression in curr_turn.program().expressions:
        op = expression.op
        if isinstance(op, CallLikeOp) and op.name == DataflowFn.Abandon.value:
            domain = curr_constraint_types[expression.id]
            slot_names_to_delete = [
                slot_name
                for slot_name in curr_belief_dict
                if slot_name.startswith(domain)
            ]
            for slot_name in slot_names_to_delete:
                del curr_belief_dict[slot_name]

        if not isinstance(op, BuildStructOp):
            continue
        if not op.op_schema.startswith("Constraint"):
            continue
        if not op.op_fields:
            continue

        # Domain independently inferred from constraint type
        domain = get_constraint_schema(op.op_schema).lower()
        for i, field in enumerate(op.op_fields):
            if field == "constraint":
                continue
            val = curr_values[expression.arg_ids[i]]

            slot_fullname = domain + "-" + field

            if val is None:
                if slot_fullname in curr_belief_dict:
                    del curr_belief_dict[slot_fullname]
            else:
                curr_belief_dict[slot_fullname] = val

    return curr_belief_dict


def execute_program_for_turn(
    curr_turn: Turn,
    salience_model: SalienceModelBase,
    last_execution_trace: ExecutionTrace,
    last_belief_dict: Optional[Dict[str, str]],
) -> Tuple[Dict[str, str], PartialExecutionResult]:
    """Executes the program for a turn."""
    partial_execution_result = PartialExecutionResult(
        values=dict(), constraints=dict(), refer_calls=dict(), slot_values=dict(),
    )
    for expression in curr_turn.program().expressions:
        partial_execution_result = execute_expression(
            expression=expression,
            salience_model=salience_model,
            last_execution_trace=last_execution_trace,
            last_partial_execution_result=partial_execution_result,
        )

    curr_belief_dict = update_belief_state(
        curr_turn=curr_turn,
        curr_values=partial_execution_result.values,
        curr_constraint_types=partial_execution_result.constraints,
        last_belief_dict=last_belief_dict,
    )

    return curr_belief_dict, partial_execution_result


def update_execution_trace(
    last_execution_trace: ExecutionTrace,
    partial_execution_result: PartialExecutionResult,
) -> ExecutionTrace:
    execution_trace = copy.deepcopy(last_execution_trace)
    for slot_name, slot_values in partial_execution_result.slot_values.items():
        if slot_name not in execution_trace.slot_values:
            execution_trace.slot_values[slot_name] = []
        execution_trace.slot_values[slot_name].extend(slot_values)
    return execution_trace


def execute_programs_for_dialogue(
    dialogue: Dialogue,
    salience_model: SalienceModelBase,
    no_revise: bool,
    cheating_mode: str,
    cheating_execution_results: Optional[Dict[int, CompleteExecutionResult]] = None,
) -> Tuple[List[CompleteExecutionResult], List[int]]:
    """Executes the programs turn-by-turn in a dialogue.

    If cheating_mode is "never", we do not use cheating_execution_results, and the execution trace is dynamically
    constructed turn by turn.
    If cheating_mode is "always", we always use cheating_execution_results as the context execution trace when executing
    the express at a turn.
    If cheating_mode is "dynamic", we only use cheating_execution_results when the actual execution result doesn't match
    the cheating execution result.
    """
    if cheating_mode == "never":
        pass
    elif cheating_mode == "always":
        assert cheating_execution_results is not None
    elif cheating_mode == "dynamic":
        assert cheating_execution_results is not None
    else:
        raise ValueError(f"Unknown cheating mode: {cheating_mode}")

    if cheating_execution_results is not None:
        assert len(cheating_execution_results) == len(dialogue.turns)

    complete_execution_results: List[CompleteExecutionResult] = []
    cheating_turn_indices: List[int] = []
    last_belief_dict: Dict[str, str] = {}
    last_execution_trace: ExecutionTrace = ExecutionTrace(slot_values={})
    curr_turn: Turn
    for curr_turn in dialogue.turns:
        try:
            belief_dict, partial_execution_result = execute_program_for_turn(
                curr_turn=curr_turn,
                salience_model=salience_model,
                last_execution_trace=last_execution_trace,
                last_belief_dict=None if no_revise else last_belief_dict,
            )
        except Exception as e:  # pylint: disable=broad-except
            belief_dict = last_belief_dict
            partial_execution_result = PartialExecutionResult(
                values=dict(),
                constraints=dict(),
                refer_calls=dict(),
                slot_values=dict(),
            )
            print(
                f"could not execute program in {dialogue.dialogue_id} {curr_turn.turn_index}: {e}",
            )

        execution_trace = update_execution_trace(
            last_execution_trace=last_execution_trace,
            partial_execution_result=partial_execution_result,
        )

        complete_execution_results.append(
            CompleteExecutionResult(
                dialogue_id=dialogue.dialogue_id,
                turn_index=curr_turn.turn_index,
                belief_dict=copy.deepcopy(belief_dict),
                execution_trace=copy.deepcopy(execution_trace),
                partial_execution_result=partial_execution_result,
            )
        )

        if cheating_mode == "never":
            last_execution_trace = execution_trace
            last_belief_dict = belief_dict
        elif cheating_mode == "always":
            cheating_execution_result = cheating_execution_results.get(
                curr_turn.turn_index
            )
            last_execution_trace = cheating_execution_result.execution_trace
            last_belief_dict = cheating_execution_result.belief_dict
            cheating_turn_indices.append(curr_turn.turn_index)
        elif cheating_mode == "dynamic":
            cheating_execution_result = cheating_execution_results.get(
                curr_turn.turn_index
            )
            if belief_dict == cheating_execution_result.belief_dict:
                last_execution_trace = execution_trace
                last_belief_dict = belief_dict
            else:
                last_execution_trace = cheating_execution_result.execution_trace
                last_belief_dict = cheating_execution_result.belief_dict
                cheating_turn_indices.append(curr_turn.turn_index)

    return complete_execution_results, cheating_turn_indices


def analyze_cheating_report(
    cheating_report_file: str, cheating_stats_file: str
) -> None:
    num_dialogues = 0
    num_cheating_turns = []
    pct_cheating_turns = []
    num_turns_before_first_cheating = []
    for line in open(cheating_report_file):
        num_dialogues += 1
        stats = json.loads(line.strip())
        num_cheating_turns.append(stats["numCheatingTurns"])
        pct_cheating_turns.append(stats["pctCheatingTurns"])
        cheating_turn_indices = stats["cheatingTurnIndices"]
        if cheating_turn_indices:
            start_turn_index = stats["startTurnIndex"]
            num_turns_before_first_cheating.append(
                cheating_turn_indices[0] - start_turn_index
            )
        else:
            num_turns_before_first_cheating.append(stats["numTurns"])

    average_num_cheating_turns = np.mean(num_cheating_turns)
    average_pct_cheating_turns = np.mean(pct_cheating_turns)
    average_num_turns_before_first_cheating = np.mean(num_turns_before_first_cheating)

    with open(cheating_stats_file, "w") as fp:
        fp.write(
            json.dumps(
                {
                    "numDialogues": num_dialogues,
                    "aveNumCheatingTurns": average_num_cheating_turns,
                    "avePctCheatingTurns": average_pct_cheating_turns,
                    "aveNumTurnsBeforeFirstCheating": average_num_turns_before_first_cheating,
                },
                indent=2,
            )
        )
        fp.write("\n")


def main(
    dialogues_file: str,
    no_refer: bool,
    no_revise: bool,
    cheating_mode: str,
    cheating_execution_results_file: Optional[str],
    outbase: str,
) -> Tuple[str, str, str]:
    salience_model: SalienceModelBase
    if no_refer:
        salience_model = DummySalienceModel()
    else:
        salience_model = VanillaSalienceModel()

    cheating_execution_results_lookup = None
    if cheating_execution_results_file is not None:
        cheating_execution_results_lookup = load_jsonl_file_and_build_lookup(
            data_jsonl=cheating_execution_results_file,
            cls=CompleteExecutionResult,
            primary_key_getter=lambda x: x.dialogue_id,
            secondary_key_getter=lambda x: x.turn_index,
        )

    complete_execution_results_file = outbase + ".execution_results.jsonl"
    cheating_report_file = outbase + ".cheating_report.jsonl"
    complete_execution_results_fp = open(complete_execution_results_file, "w")
    cheating_report_fp = open(cheating_report_file, "w")

    for dialogue in load_jsonl_file(
        data_jsonl=dialogues_file, cls=Dialogue, unit=" dialogues"
    ):
        if cheating_execution_results_lookup is None:
            cheating_execution_results = None
        else:
            cheating_execution_results = cheating_execution_results_lookup.get(
                dialogue.dialogue_id
            )
            assert cheating_execution_results is not None

        (
            complete_execution_results,
            cheating_turn_indices,
        ) = execute_programs_for_dialogue(
            dialogue=dialogue,
            salience_model=salience_model,
            no_revise=no_revise,
            cheating_mode=cheating_mode,
            cheating_execution_results=cheating_execution_results,
        )

        for complete_execution_result in complete_execution_results:
            complete_execution_results_fp.write(jsons.dumps(complete_execution_result))
            complete_execution_results_fp.write("\n")

        num_total_turns = len(dialogue.turns)
        assert (
            dialogue.turns[-1].turn_index - dialogue.turns[0].turn_index + 1
            == num_total_turns
        )
        num_cheating_turns = len(cheating_turn_indices)
        cheating_report_fp.write(
            json.dumps(
                {
                    "dialogueId": dialogue.dialogue_id,
                    "startTurnIndex": dialogue.turns[0].turn_index,
                    "numTurns": num_total_turns,
                    "cheatingTurnIndices": cheating_turn_indices,
                    "numCheatingTurns": num_cheating_turns,
                    "pctCheatingTurns": num_cheating_turns / num_total_turns,
                }
            )
        )
        cheating_report_fp.write("\n")
    complete_execution_results_fp.close()
    cheating_report_fp.close()

    cheating_stats_file = outbase + ".cheating_stats.json"
    analyze_cheating_report(cheating_report_file, cheating_stats_file)

    return complete_execution_results_file, cheating_report_file, cheating_stats_file


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--dialogues_file", help="the file containing dialogues to be executed",
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
        "--cheating_mode",
        choices=["never", "always", "dynamic"],
        help="the way to use the cheating CompleteExecutionResult during execution",
    )
    argument_parser.add_argument(
        "--cheating_execution_results_file",
        default=None,
        help="the optional cheating CompleteExecutionResult file",
    )
    argument_parser.add_argument("--outbase", help="output files basename")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    print("Semantic Machines\N{TRADE MARK SIGN} software.")
    main(
        dialogues_file=args.dialogues_file,
        no_refer=args.no_refer,
        no_revise=args.no_revise,
        cheating_mode=args.cheating_mode,
        cheating_execution_results_file=args.cheating_execution_results_file,
        outbase=args.outbase,
    )
