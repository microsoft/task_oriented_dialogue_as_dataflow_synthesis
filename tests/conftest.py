#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from os.path import dirname, join

import pytest


@pytest.fixture(scope="module")
def data_dir() -> str:
    base_dir = dirname(__file__)
    return join(base_dir, "data")
