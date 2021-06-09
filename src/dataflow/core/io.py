#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Iterator, Type, TypeVar, cast

import jsons
from tqdm import tqdm

# type of a dataclass object
_TDatum = TypeVar("_TDatum")
# type of the primary key for _TDatum
_TPrimaryKey = TypeVar("_TPrimaryKey")
# type of the secondary key for _TDatum
_TSecondaryKey = TypeVar("_TSecondaryKey")


def load_jsonl_file(
    data_jsonl: str, cls: Type[_TDatum], unit: str = " items", verbose: bool = True
) -> Iterator[_TDatum]:
    """Loads a jsonl file and yield the deserialized dataclass objects."""
    if verbose:
        desc = f"Reading {cls} from {data_jsonl}"
    else:
        desc = None
    with open(data_jsonl, encoding="utf-8") as fp:
        for line in tqdm(
            fp, desc=desc, unit=unit, dynamic_ncols=True, disable=not verbose
        ):
            yield jsons.loads(line.strip(), cls=cls)


def save_jsonl_file(
    data: Iterable[_TDatum], data_jsonl: str, remove_null: bool = True
) -> None:
    """Dumps dataclass objects into a jsonl file."""
    with open(data_jsonl, "w") as fp:
        for datum in data:
            datum_dict = cast(Dict[str, Any], jsons.dump(datum))
            if remove_null:
                fp.write(json.dumps(remove_null_fields_in_dict(datum_dict)))
            else:
                fp.write(json.dumps(datum_dict))
            fp.write("\n")


def load_jsonl_file_and_build_lookup(
    data_jsonl: str,
    cls: Type[_TDatum],
    primary_key_getter: Callable[[_TDatum], _TPrimaryKey],
    secondary_key_getter: Callable[[_TDatum], _TSecondaryKey],
    unit: str = " items",
    verbose: bool = True,
) -> Dict[_TPrimaryKey, Dict[_TSecondaryKey, _TDatum]]:
    """Loads a jsonl file of serialized dataclass objects and returns the lookup with a primary key and a secondary key."""
    if verbose:
        desc = f"Reading {cls} from {data_jsonl}"
    else:
        desc = None
    data_lookup: Dict[_TPrimaryKey, Dict[_TSecondaryKey, _TDatum]] = defaultdict(dict)
    with open(data_jsonl) as fp:
        for line in tqdm(
            fp, desc=desc, unit=unit, dynamic_ncols=True, disable=not verbose
        ):
            datum = jsons.loads(line.strip(), cls)
            primary_key = primary_key_getter(datum)
            if primary_key not in data_lookup:
                data_lookup[primary_key] = {}
            data_lookup[primary_key][secondary_key_getter(datum)] = datum
    return data_lookup


def remove_null_fields_in_dict(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Removes null fields in the dict object."""
    return {
        key: remove_null_fields_in_dict(val) if isinstance(val, dict) else val
        for key, val in raw_dict.items()
        if val is not None
    }
