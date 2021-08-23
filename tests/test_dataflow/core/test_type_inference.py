from typing import Dict

from dataflow.core.definition import Definition
from dataflow.core.lispress import lispress_to_program, parse_lispress
from dataflow.core.program import TypeName
from dataflow.core.type_inference import infer_types, TypeInferenceError
import pytest


def _do_inference_test(expr: str, expected: str, library: Dict[str, Definition]):
    lispress = parse_lispress(expr)
    program, _ = lispress_to_program(lispress, 0)
    res = infer_types(program, library)
    (expected_program, unused_idx) = lispress_to_program(parse_lispress(expected), 0,)
    return expected_program, res


SIMPLE_PLUS_LIBRARY = {
    "+": Definition("+", ["T"], [TypeName("T"), TypeName("T")], TypeName("T")),
    "plusLong": Definition(
        "+", [], [TypeName("Long"), TypeName("Long")], TypeName("Long")
    ),
    "single_element_list": Definition("single_element_list", ["T"], [TypeName("T")], TypeName("List", [TypeName("T")])),
}


def test_simple():
    expected_program, res = _do_inference_test(
        "(+ (plusLong 3L 1L) 2L)",
        "^Long (^(Long) + ^Long (plusLong ^Long 3L ^Long 1L) ^Long 2L)",
        SIMPLE_PLUS_LIBRARY,
    )
    assert res == expected_program

    expected_program, res = _do_inference_test(
        "(+ 1 2)",
        "^Number (^(Number) + ^Number 1 ^Number 2)",
        SIMPLE_PLUS_LIBRARY,
    )
    assert res == expected_program


def test_types_disagree():
    with pytest.raises(TypeInferenceError):
        _do_inference_test(
            "^Number (plusLong 3L 1)",
            "^Number (plusLong 3L 1)",
            SIMPLE_PLUS_LIBRARY,
        )


def test_ascription_disagrees():
    with pytest.raises(TypeInferenceError):
        _do_inference_test(
            "^Number (plusLong 3L 1L)",
            "^Number (plusLong 3L 1L)",
            SIMPLE_PLUS_LIBRARY,
        )


def test_let():
    expected_program, res = _do_inference_test(
        "(let (x (+ 1L 2L)) (+ x x))",
        "(let (x ^Long (^(Long) + ^Long 1L ^Long 2L)) ^Long (^(Long) + x x))",
        SIMPLE_PLUS_LIBRARY,
    )
    assert res == expected_program


def test_multi_let():
    expected_program, res = _do_inference_test(
        "(let (a 1L b 2L x (+ a b)) (+ x x))",
        "(let (a ^Long 1L b ^Long 2L x ^Long (^(Long) + a b)) ^Long (^(Long) + x x))",
        SIMPLE_PLUS_LIBRARY,
    )
    assert res == expected_program


def test_parameterized():
    expected_program, res = _do_inference_test(
        "(single_element_list 1)",
        "^(List Number) (^(Number) single_element_list ^Number 1)",
        SIMPLE_PLUS_LIBRARY,
    )
    assert res == expected_program

    expected_program, res = _do_inference_test(
        '(single_element_list "5")',
        '^(List String) (^(String) single_element_list ^String "5")',
        SIMPLE_PLUS_LIBRARY,
    )
    assert res == expected_program
