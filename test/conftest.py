# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pytest configuration for Circuit Knitting Toolbox"""

import pytest


# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option


# pylint: disable=missing-function-docstring
def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="skip slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="skipping slow test, as --skip-slow was provided")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
