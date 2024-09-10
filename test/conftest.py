# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pytest configuration for the Qiskit addon for circuit cutting"""

import pytest


# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option


# pylint: disable=missing-function-docstring
def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--coverage",
        action="store_true",
        default=False,
        help="skip tests that should not be used for calculating coverage",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "skipforcoverage: skip test during coverage run")


def pytest_collection_modifyitems(config, items):
    flags = (
        (
            "--run-slow",
            "slow",
            False,
            "skipping slow test, as --run-slow was not provided",
        ),
        (
            "--coverage",
            "skipforcoverage",
            True,
            "deliberately skipping, as --coverage was provided",
        ),
    )
    for option, keyword, skip_when, reason in flags:
        if config.getoption(option) is skip_when:
            marker = pytest.mark.skip(reason=reason)
            for item in items:
                if keyword in item.keywords:
                    item.add_marker(marker)
