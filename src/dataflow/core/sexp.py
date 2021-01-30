#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from enum import Enum
from typing import List, Union

LEFT_PAREN = "("
RIGHT_PAREN = ")"
ESCAPE = "\\"
DOUBLE_QUOTE = '"'

# we unwrap Symbols into strings for convenience
Sexp = Union[str, List["Sexp"]]  # type: ignore # Recursive type


class QuoteState(Enum):
    """Whether a char is inside double quotes or not during parsing"""

    Inside = True
    Outside = False

    def flipped(self) -> "QuoteState":
        return QuoteState(not self.value)


def _split_respecting_quotes(s: str) -> List[str]:
    """Splits `s` on whitespace that is not inside double quotes."""
    result = []
    marked_idx = 0
    state = QuoteState.Outside
    for idx, ch in enumerate(s):
        if ch == DOUBLE_QUOTE and (idx < 1 or s[idx - 1] != ESCAPE):
            if state == QuoteState.Outside:
                result.extend(s[marked_idx:idx].strip().split())
                marked_idx = idx
            else:
                result.append(s[marked_idx : idx + 1])  # include the double quote
                marked_idx = idx + 1
            state = state.flipped()
    assert state == QuoteState.Outside, f"Mismatched double quotes: {s}"
    if marked_idx < len(s):
        result.extend(s[marked_idx:].strip().split())
    return result


def parse_sexp(sexp_string: str, clean_singletons=False) -> Sexp:
    """ Parses an S-expression from a string """
    sexp_string = sexp_string.strip()
    # handle some special cases
    if sexp_string == "":
        return []
    if sexp_string[-1] == ";":
        sexp_string = sexp_string[:-1]
    # find and group top-level parentheses (that are not inside quoted strings)
    num_open_brackets = 0
    open_bracket_idxs = []
    close_bracket_idxs = []
    result: List[Sexp] = []
    state = QuoteState.Outside
    for index, ch in enumerate(sexp_string):
        if ch == DOUBLE_QUOTE and (index < 1 or sexp_string[index - 1] != ESCAPE):
            state = state.flipped()
        if ch == LEFT_PAREN and state == QuoteState.Outside:
            num_open_brackets += 1
            if num_open_brackets == 1:
                open_bracket_idxs.append(index)
        elif ch == RIGHT_PAREN and state == QuoteState.Outside:
            num_open_brackets -= 1
            if num_open_brackets == 0:
                close_bracket_idxs.append(index)

    assert len(open_bracket_idxs) == len(
        close_bracket_idxs
    ), f"Mismatched parentheses: {sexp_string}"
    assert state == QuoteState.Outside, f"Mismatched double quotes: {sexp_string}"

    start = 0
    for index, (open_bracket_idx, close_bracket_idx) in enumerate(
        zip(open_bracket_idxs, close_bracket_idxs)
    ):
        if start < open_bracket_idx:
            preparen = sexp_string[start:open_bracket_idx].strip()
            if preparen != "":
                tokens = _split_respecting_quotes(preparen)
                result.extend(tokens)
        result.append(
            parse_sexp(
                sexp_string[open_bracket_idx + 1 : close_bracket_idx],
                clean_singletons=clean_singletons,
            )
        )
        start = close_bracket_idx + 1

    if start < len(sexp_string):
        # tokens after the last ')'
        postparen = sexp_string[start:].strip()
        if postparen != "":
            tokens = _split_respecting_quotes(postparen)
            result.extend(tokens)

    if len(result) == 2 and result[-1] == ";":
        sexp_tmp = result[0]
    else:
        sexp_tmp = result

    # special-case top-level values because they need an extra level
    # of parens in the Sexp:
    if isinstance(result, list) and len(result) >= 1 and result[0] == "#":
        return [result]
    if clean_singletons and len(sexp_tmp) == 1:
        return sexp_tmp[0]
    return sexp_tmp


def sexp_to_str(sexp: Sexp) -> str:
    """ Generates string representation from S-expression """
    if isinstance(sexp, list):
        return "(" + " ".join(sexp_to_str(f) for f in sexp) + ")"
    else:
        return sexp


def flatten(form: Sexp) -> List[str]:
    return (
        [form]
        if not isinstance(form, list)
        else [s for subexp in form for s in flatten(subexp)]
    )
