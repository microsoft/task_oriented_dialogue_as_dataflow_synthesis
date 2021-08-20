from dataflow.core.definition import Definition
from dataflow.core.lispress import parse_lispress, lispress_to_program
from dataflow.core.program import TypeName
from dataflow.core.type_inference import infer_types


def test_simple():
    lispress_str = "(+ (plusLong 3L 1L) 2L)"
    lispress = parse_lispress(lispress_str)
    program, _ = lispress_to_program(lispress, 0)
    library = {
        "+": Definition("+", ["T"], [TypeName("T"), TypeName("T")], TypeName("T")),
        "plusLong": Definition("+",
                               [],
                               [TypeName("Long"), TypeName("Long")],
                               TypeName("Long")),
        "Long": Definition("Long", [], [TypeName("Unit")], TypeName("Long")),
    }

    res = infer_types(program, library)
    print(res)
