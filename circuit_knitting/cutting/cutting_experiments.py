# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for evaluating circuit cutting experiments."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliList

from .qpd import WeightType
from .cutting_evaluation import _generate_cutting_experiments


def generate_cutting_experiments(
    circuits: QuantumCircuit | dict[str | int, QuantumCircuit],
    observables: PauliList | dict[str | int, PauliList],
    num_samples: int | float,
) -> tuple[
    list[QuantumCircuit] | dict[str | int, list[QuantumCircuit]],
    list[tuple[float, WeightType]],
]:
    """
    Generate cutting subexperiments and their associated weights.

    If the input, ``circuits``, is a :class:`QuantumCircuit` instance, the
    output subexperiments will be contained within a 1D array, and ``observables`` is
    expected to be a :class:`PauliList` instance.

    If the input circuit and observables are specified by dictionaries with partition labels
    as keys, the output subexperiments will be returned as a dictionary which maps a
    partition label to to a 1D array containing the subexperiments associated with that partition.

    In both cases, the subexperiment lists are ordered as follows:
        :math:`[sample_{0}observable_{0}, sample_{0}observable_{1}, ..., sample_{0}observable_{N}, ..., sample_{M}observable_{N}]`

    The weights will always be returned as a 1D array -- one weight for each unique sample.

    Args:
        circuits: The circuit(s) to partition and separate
        observables: The observable(s) to evaluate for each unique sample
        num_samples: The number of samples to draw from the quasi-probability distribution. If set
            to infinity, the weights will be generated rigorously rather than by sampling from
            the distribution.
    Returns:
        A tuple containing the cutting experiments and their associated weights.
        If the input circuits is a :class:`QuantumCircuit` instance, the output subexperiments
        will be a sequence of circuits -- one for every unique sample and observable. If the
        input circuits are represented as a dictionary keyed by partition labels, the output
        subexperiments will also be a dictionary keyed by partition labels and containing
        the subexperiments for each partition.
        The weights are always a sequence of length-2 tuples, where each tuple contains the
        weight and the :class:`WeightType`. Each weight corresponds to one unique sample.

    Raises:
        ValueError: ``num_samples`` must either be at least one.
        ValueError: ``circuits`` and ``observables`` are incompatible types
        ValueError: :class:`SingleQubitQPDGate` instances must have their cut ID
            appended to the gate label so they may be associated with other gates belonging
            to the same cut.
        ValueError: :class:`SingleQubitQPDGate` instances are not allowed in unseparated circuits.
    """
    subexperiments, weights, _ = _generate_cutting_experiments(
        circuits, observables, num_samples
    )
    return subexperiments, weights
