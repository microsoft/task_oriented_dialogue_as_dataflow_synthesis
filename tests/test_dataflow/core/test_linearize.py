#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import json

from dataflow.core.linearize import (
    lispress_to_seq,
    seq_to_lispress,
    sexp_to_seq,
    to_canonical_form,
)
from dataflow.core.lispress import parse_lispress, render_compact, unnest_line
from dataflow.core.program import Expression, ValueOp


def test_sexp_to_seq_is_invertible():
    sexps = [
        [],
        ["a"],
        ["a", ["b", "c"], "d", [[["e"]]]],
        [["a", "b"], [["c"], "d"]],
    ]
    for orig in sexps:
        seq = sexp_to_seq(orig)
        result = seq_to_lispress(seq)
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


def test_linearized_roundtrip():
    """Round-trip tests for s-expression formatter and deformatter."""
    data = [
        ('#(String "singleToken")', '# ( String " singleToken " )'),
        ('#(String "multiple tokens")', '# (  String " multiple tokens " )'),
        # real data
        (
            '((mapGet #(Path "fare.fare_id") (clobberRevise (getSalient (actionIntensionConstraint)) ('
            "Constraint[Constraint[flight]]) (Constraint "
            ':type (?= #(String "fare"))))))',
            '( ( mapGet # ( Path " fare.fare_id " ) ( clobberRevise ( getSalient ( actionIntensionConstraint ) ) '
            "( Constraint[Constraint[flight]] ) ( "
            'Constraint :type ( ?= # ( String " fare " ) ) ) ) ) )',
        ),
    ]

    for raw_sexp, formatted_sexp in data:
        assert lispress_to_seq(parse_lispress(raw_sexp)) == formatted_sexp.split()
        assert render_compact(seq_to_lispress(formatted_sexp.split())) == raw_sexp


def test_meta():
    assert lispress_to_seq(
        parse_lispress("(refer (^(Dynamic) ActionIntensionConstraint))")
    ) == [
        "(",
        "refer",
        "(",
        "^",
        "(",
        "Dynamic",
        ")",
        "ActionIntensionConstraint",
        ")",
        ")",
    ]


def test_meta_to_canonical():
    s = """( Yield ( Execute ( ReviseConstraint ( refer ( ^ ( Dynamic ) roleConstraint ( Path.apply "output" ) ) ) ( ^ ( Event ) ConstraintTypeIntension ) ( Event.showAs_? ( ?= ( ShowAsStatus.OutOfOffice ) ) ) ) ) )"""
    assert (
        to_canonical_form(s)
        == """(Yield (Execute (ReviseConstraint (refer (^(Dynamic) roleConstraint (Path.apply "output"))) (^(Event) ConstraintTypeIntension) (Event.showAs_? (?= (ShowAsStatus.OutOfOffice))))))"""
    )


def test_quoted():
    s = """( foo " bar " )"""
    assert to_canonical_form(s) == """(foo "bar")"""
