#!/usr/bin/env python3
# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import re
import configparser

import toml
import fire


def mapfunc_dev(dep):
    """Load the development version(s) of certain Qiskit-related packages"""
    # https://peps.python.org/pep-0440/#direct-references
    return re.sub(
        r"^(qiskit-(?:terra|ibm-runtime)).*$",
        r"\1 @ git+https://github.com/Qiskit/\1.git",
        dep,
    )


def mapfunc_min(dep):
    """Set each dependency to its minimum version"""
    return re.sub(r"[>~]=", r"==", dep)


def inplace_map(fun, lst: list):
    """In-place version of Python's `map` function"""
    for i, x in enumerate(lst):
        lst[i] = fun(x)


def process_dependencies_in_place(d: dict, mapfunc):
    """Given a parsed `pyproject.toml`, process dependencies according to `mapfunc`"""
    proj = d["project"]

    try:
        deps = proj["dependencies"]
    except KeyError:
        pass  # no dependencies; that's unusual, but fine.
    else:
        inplace_map(mapfunc, deps)

    try:
        opt_deps = proj["optional-dependencies"]
    except KeyError:
        pass  # no optional dependencies; that's fine.
    else:
        for dependencies_list in opt_deps.values():
            inplace_map(mapfunc, dependencies_list)

    try:
        build_system = d["build-system"]
    except KeyError:
        pass
    else:
        try:
            build_system_requires = build_system["requires"]
        except KeyError:
            pass
        else:
            inplace_map(mapfunc, build_system_requires)


class CLI:
    """Command-line interface class for Fire"""

    def get_tox_minversion(self):
        """Extract tox minversion from `tox.ini`"""
        config = configparser.ConfigParser()
        config.read("tox.ini")
        print(config["tox"]["minversion"])

    def pin_dependencies(self, strategy, inplace: bool = False):
        """Pin the dependencies in `pyproject.toml` according to `strategy`"""
        mapfunc = {
            "dev": mapfunc_dev,
            "min": mapfunc_min,
        }[strategy]

        with open("pyproject.toml") as f:
            d = toml.load(f)
        process_dependencies_in_place(d, mapfunc)

        # Modify pyproject.toml so hatchling will allow direct references
        # as dependencies.
        d.setdefault("tool", {}).setdefault("hatch", {}).setdefault("metadata", {})[
            "allow-direct-references"
        ] = True

        if inplace:
            with open("pyproject.toml", "w") as f:
                toml.dump(d, f)
        else:
            print(toml.dumps(d))


if __name__ == "__main__":
    fire.Fire(CLI)
