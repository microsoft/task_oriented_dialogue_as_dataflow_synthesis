from dataclasses import dataclass, field
from typing import List, Optional
from dataflow.core.program import TypeName


@dataclass(frozen=True)
class Definition:
    name: str
    type_args: List[str]
    args: List[TypeName]
    type: TypeName

