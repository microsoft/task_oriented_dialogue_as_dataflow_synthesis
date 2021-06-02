#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from enum import Enum
from typing import List, Union

LEFT_PAREN = "("
RIGHT_PAREN = ")"
ESCAPE = "\\"
DOUBLE_QUOTE = '"'
META = "^"
READER = "#"

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


def parse_sexp(s: str) -> Sexp:
    offset = 0

    # eoi = end of input
    def is_eoi():
        nonlocal offset
        return offset == len(s)

    def peek():
        nonlocal offset
        return s[offset]

    def next_char():
        # pylint: disable=used-before-assignment
        nonlocal offset
        cn = s[offset]
        offset += 1
        return cn

    def skip_whitespace():
        while (not is_eoi()) and peek().isspace():
            next_char()

    def skip_then_peek():
        skip_whitespace()
        return peek()

    def read() -> Sexp:
        skip_whitespace()
        c = next_char()
        if c == LEFT_PAREN:
            return read_list()
        elif c == DOUBLE_QUOTE:
            return read_string()
        elif c == META:
            meta = read()
            expr = read()
            return [META, meta, expr]
        elif c == READER:
            return [READER, read()]
        else:
            out_inner = ""
            if c != "\\":
                out_inner += c

            # TODO: is there a better loop idiom here?
            if not is_eoi():
                next_c = peek()
                escaped = c == "\\"
                while (not is_eoi()) and (
                    escaped or not _is_beginning_control_char(next_c)
                ):
                    if (not escaped) and next_c == "\\":
                        next_char()
                        escaped = True
                    else:
                        out_inner += next_char()
                        escaped = False
                    if not is_eoi():
                        next_c = peek()
            return out_inner

    def read_list():
        out_list = []
        while skip_then_peek() != RIGHT_PAREN:
            out_list.append(read())
        next_char()
        return out_list

    def read_string():
        out_str = ""
        while peek() != '"':
            c_string = next_char()
            out_str += c_string
            if c_string == "\\":
                out_str += next_char()
        next_char()
        return f'"{out_str}"'

    out = read()
    skip_whitespace()
    assert offset == len(
        s
    ), f"Failed to exhaustively parse {s}, maybe you are missing a close paren?"
    return out


def _is_beginning_control_char(nextC):
    return (
        nextC.isspace()
        or nextC == LEFT_PAREN
        or nextC == RIGHT_PAREN
        or nextC == DOUBLE_QUOTE
        or nextC == READER
        or nextC == META
    )


def sexp_to_str(sexp: Sexp) -> str:
    """ Generates string representation from S-expression """
    # Note that some of this logic is repeated in lispress.render_pretty
    if isinstance(sexp, list):
        if len(sexp) == 3 and sexp[0] == META:
            (_meta, type_expr, underlying_expr) = sexp
            return META + sexp_to_str(type_expr) + " " + sexp_to_str(underlying_expr)
        elif len(sexp) == 2 and sexp[0] == READER:
            (_reader, expr) = sexp
            return READER + sexp_to_str(expr)
        else:
            return "(" + " ".join(sexp_to_str(f) for f in sexp) + ")"
    else:
        if sexp.startswith('"') and sexp.endswith('"'):
            return sexp
        else:
            return _escape_symbol(sexp)


def _escape_symbol(symbol: str) -> str:
    out = []
    for c in symbol:
        if _is_beginning_control_char(c):
            out.append("\\")
        out.append(c)
    return "".join(out)
