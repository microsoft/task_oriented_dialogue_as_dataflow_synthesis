#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import json

from dataflow.core.linearize import (
    program_to_seq,
    seq_to_program,
    seq_to_sexp,
    sexp_deformatter,
    sexp_formatter,
    sexp_to_seq,
)
from dataflow.core.lispress import unnest_line
from dataflow.core.program import Expression, Program, ValueOp
from dataflow.core.sexp import flatten


def test_plan_to_seq_strict():
    """Strict tests the conversion between the Program and the linearized plan."""
    original_program = Program(
        expressions=[
            Expression(
                id="[1]",
                op=ValueOp(
                    value=json.dumps(
                        {
                            "schema": "Placeholder[List[CreateCommitEvent]]",
                            "underlying": {
                                "error": {
                                    "schema": "EmptySlot",
                                    "underlying": {
                                        "slotId": {
                                            "schema": "String",
                                            "underlying": "255905761",
                                        }
                                    },
                                }
                            },
                        }
                    )
                ),
            ),
        ]
    )
    expected_linearized_plan = (
        "( # ( Placeholder[List[CreateCommitEvent]] "
        '{ "error" : { "schema" : "EmptySlot" , '
        '"underlying" : { "slotId" : { "schema" : "String" , "underlying" : " 255905761 " } } '
        "} } "
        ") )"
    )

    linearized_plan_tokens = program_to_seq(program=original_program)
    assert all(" " not in tok for tok in flatten(linearized_plan_tokens))
    assert " ".join(linearized_plan_tokens) == expected_linearized_plan

    program, _ = seq_to_program(seq=linearized_plan_tokens, idx=0)
    assert program == original_program


def test_sexp_to_seq_is_invertible():
    sexps = [
        [],
        ["a"],
        ["a", ["b", "c"], "d", [[["e"]]]],
        [["a", "b"], [["c"], "d"]],
    ]
    for orig in sexps:
        seq = sexp_to_seq(orig)
        result = seq_to_sexp(seq)
        assert result == orig


def test_unnest_line():
    line = ["#", ["String", '"pizza', "hut", 'fenditton"']]
    expected_expression = Expression(
        id="[1]",
        op=ValueOp(
            value=json.dumps(dict(schema="String", underlying="pizza hut fenditton"))
        ),
    )

    expressions, _, _, _ = unnest_line(line, 0, ())
    assert len(expressions) == 1
    assert expressions[0] == expected_expression


def test_sexp_formatter_and_deformatter():
    """Round-trip tests for s-expression formatter and deformatter."""
    data = [
        ("", ""),
        ('# ( String "singleToken" )', '# ( String " singleToken " )'),
        ('# ( String "multiple tokens" )', '# (  String " multiple tokens " )'),
        # these two are not valid value s-expressions, so they are not formatted
        ('# ( String "open quote only )', '# (  String "open quote only )'),
        ('# ( String closing quote only" )', '# ( String closing quote only" )',),
        # multiple value s-expressions
        (
            '# ( String "singleToken" ) # ( String "multiple tokens" )',
            '# ( String " singleToken " ) # ( String " multiple tokens " )',
        ),
        # real data
        (
            '( ( mapGet ( # ( Path "fare.fare_id" ) ) ( clobberRevise ( getSalient ( actionIntensionConstraint ) ) ( '
            "Constraint[Constraint[flight]] ) ( Constraint "
            ':type ( ?= ( # ( String "fare" ) ) ) ) ) ) )',
            '( ( mapGet ( # ( Path " fare.fare_id " ) ) ( clobberRevise ( getSalient ( actionIntensionConstraint ) ) '
            "( Constraint[Constraint[flight]] ) ( "
            'Constraint :type ( ?= ( # ( String " fare " ) ) ) ) ) ) )',
        ),
    ]

    for raw_sexp, formatted_sexp in data:
        assert sexp_formatter(raw_sexp.split()) == formatted_sexp.split()
        assert sexp_deformatter(formatted_sexp.split()) == raw_sexp.split()
