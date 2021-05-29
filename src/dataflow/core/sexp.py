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

    def isEOI():
        nonlocal offset
        return offset == len(s)

    def peek():
        nonlocal offset
        return s[offset]

    def nextChar():
        # pylint: disable=used-before-assignment
        nonlocal offset
        cn = s[offset]
        offset += 1
        return cn

    def skipWhitespace():
        while (not isEOI()) and peek().isspace():
            nextChar()

    def skipThenPeek():
        skipWhitespace()
        return peek()

    def read() -> Sexp:
        skipWhitespace()
        c = nextChar()
        if c == LEFT_PAREN:
            return readList()
        elif c == DOUBLE_QUOTE:
            return readString()
        elif c == META:
            meta = read()
            expr = read()
            return [META, meta, expr]
        elif c == READER:
            return [READER, read()]
        else:
            outInner = ""
            if c != "\\":
                outInner += c

            # TODO: is there a better loop idiom here?
            if not isEOI():
                nextC = peek()
                escaped = c == "\\"
                while (not isEOI()) and (escaped or not _isBeginningControlChar(nextC)):
                    if (not escaped) and nextC == "\\":
                        nextChar()
                        escaped = True
                    else:
                        outInner += nextChar()
                        escaped = False
                    if not isEOI():
                        nextC = peek()
            return outInner

    def readList():
        outList = []
        while skipThenPeek() != RIGHT_PAREN:
            outList.append(read())
        nextChar()
        return outList

    def readString():
        outStr = ""
        while peek() != '"':
            cString = nextChar()
            outStr += cString
            if cString == "\\":
                outStr += nextChar()
        nextChar()
        return f'"{outStr}"'

    out = read()
    skipWhitespace()
    return out


def _isBeginningControlChar(nextC):
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
    if isinstance(sexp, list):
        if len(sexp) == 3 and sexp[0] == META:
            return META + sexp_to_str(sexp[1]) + " " + sexp_to_str(sexp[2])
        elif len(sexp) == 2 and sexp[0] == READER:
            return READER + sexp_to_str(sexp[1])
        else:
            return "(" + " ".join(sexp_to_str(f) for f in sexp) + ")"
    else:
        return sexp
