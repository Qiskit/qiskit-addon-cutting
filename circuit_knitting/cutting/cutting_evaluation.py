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

from typing import NamedTuple
from collections.abc import Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliList
from qiskit.primitives import BaseSampler, Sampler as TerraSampler, SamplerResult
from qiskit_aer.primitives import Sampler as AerSampler

from .qpd import WeightType
from .cutting_experiments import generate_cutting_experiments


class CuttingExperimentResults(NamedTuple):
    """Circuit cutting subexperiment results and sampling coefficients."""

    results: SamplerResult | dict[str | int, SamplerResult]
    coeffs: Sequence[tuple[float, WeightType]]


def execute_experiments(
    circuits: QuantumCircuit | dict[str | int, QuantumCircuit],
    subobservables: PauliList | dict[str | int, PauliList],
    num_samples: int,
    samplers: BaseSampler | dict[str | int, BaseSampler],
) -> CuttingExperimentResults:
    r"""
    Generate the sampled circuits, append the observables, and run the sub-experiments.

    Args:
        circuits: The circuit(s) resulting from decomposing nonlocal gates
        subobservables: The subobservable(s) corresponding to the circuit(s). If
            a :class:`~qiskit.circuit.QuantumCircuit` is submitted for the ``circuits`` argument,
            a :class:`~qiskit.quantum_info.PauliList` is expected; otherwise, a mapping
            from partition label to subobservables is expected.
        num_samples: The number of samples to draw from the quasiprobability distribution
        samplers: Sampler(s) on which to run the sub-experiments.

    Returns:
        - One :class:`~qiskit.primitives.SamplerResult` instance for each partition.
        - Coefficients corresponding to each unique subexperiment's contribution to the reconstructed result

    Raises:
        ValueError: The number of requested samples must be at least one.
        ValueError: The types of ``circuits`` and ``subobservables`` arguments are incompatible.
        ValueError: ``SingleQubitQPDGate``\ s are not supported in unseparable circuits.
        ValueError: The keys for the input dictionaries are not equivalent.
        ValueError: The input circuits may not contain any classical registers or bits.
        ValueError: If multiple samplers are passed, each one must be unique.
    """
    if not num_samples >= 1:
        raise ValueError("The number of requested samples must be at least 1.")

    if isinstance(circuits, dict) and not isinstance(subobservables, dict):
        raise ValueError(
            "If a partition mapping (dict[label, subcircuit]) is passed as the "
            "circuits argument, a partition mapping (dict[label, subobservables]) "
            "is expected as the subobservables argument."
        )

    if isinstance(circuits, QuantumCircuit) and isinstance(subobservables, dict):
        raise ValueError(
            "If a QuantumCircuit is passed as the circuits argument, a PauliList "
            "is expected as the subobservables argument."
        )

    if isinstance(circuits, dict):
        if circuits.keys() != subobservables.keys():
            raise ValueError(
                "The keys for the circuits and observables dicts should be equivalent."
            )
        if isinstance(samplers, dict) and circuits.keys() != samplers.keys():
            raise ValueError(
                "The keys for the circuits and samplers dicts should be equivalent."
            )

    if isinstance(samplers, dict):
        # Ensure that each sampler is unique
        collision_dict: dict[int, str | int] = {}
        for k, v in samplers.items():
            if id(v) in collision_dict:
                raise ValueError(
                    "Currently, if a samplers dict is passed to execute_experiments(), "
                    "then each sampler must be unique; however, subsystems "
                    f"{collision_dict[id(v)]} and {k} were passed the same sampler."
                )
            collision_dict[id(v)] = k

    # Ensure input Samplers can handle mid-circuit measurements
    _validate_samplers(samplers)

    # Generate the sub-experiments to run on backend
    subexperiments, coefficients = generate_cutting_experiments(
        circuits, subobservables, num_samples
    )

    # Set up subexperiments and samplers
    subexperiments_dict: dict[str | int, list[QuantumCircuit]] = {}
    if isinstance(subexperiments, list):
        subexperiments_dict = {"A": subexperiments}
    else:
        assert isinstance(subexperiments, dict)
        subexperiments_dict = subexperiments
    if isinstance(samplers, BaseSampler):
        samplers_dict = {key: samplers for key in subexperiments_dict.keys()}
    else:
        assert isinstance(samplers, dict)
        samplers_dict = samplers

    # Make sure the first two cregs in each circuit are for QPD and observable measurements
    # Run a job for each partition and collect results
    results = {}
    for label in sorted(subexperiments_dict.keys()):
        for circ in subexperiments_dict[label]:
            if (
                len(circ.cregs) != 2
                or circ.cregs[1].name != "observable_measurements"
                or circ.cregs[0].name != "qpd_measurements"
                or sum([reg.size for reg in circ.cregs]) != circ.num_clbits
            ):
                # If the classical bits/registers are in any other format than expected, the user must have
                # input them, so we can just raise this generic error in any case.
                raise ValueError(
                    "Circuits input to execute_experiments should contain no classical registers or bits."
                )
        results[label] = samplers_dict[label].run(subexperiments_dict[label]).result()

    for label, result in results.items():
        for i, metadata in enumerate(result.metadata):
            metadata["num_qpd_bits"] = len(subexperiments_dict[label][i].cregs[0])

    # If the input was a circuit, the output results should be a single SamplerResult instance
    results_out = results
    if isinstance(circuits, QuantumCircuit):
        assert len(results_out.keys()) == 1
        results_out = results[list(results.keys())[0]]

    return CuttingExperimentResults(results=results_out, coeffs=coefficients)


def _validate_samplers(samplers: BaseSampler | dict[str | int, BaseSampler]) -> None:
    """Replace unsupported statevector-based Samplers with ExactSampler."""
    if isinstance(samplers, BaseSampler):
        if (
            isinstance(samplers, AerSampler)
            and "shots" in samplers.options
            and samplers.options.shots is None
        ):
            _aer_sampler_error()
        elif isinstance(samplers, TerraSampler):
            _terra_sampler_error()

    elif isinstance(samplers, dict):
        for key, sampler in samplers.items():
            if (
                isinstance(sampler, AerSampler)
                and "shots" in sampler.options
                and sampler.options.shots is None
            ):
                _aer_sampler_error()
            elif isinstance(sampler, TerraSampler):
                _terra_sampler_error()
            elif isinstance(sampler, BaseSampler):
                continue
            else:
                _bad_samplers_error()

    else:
        _bad_samplers_error()


def _aer_sampler_error() -> None:
    raise ValueError(
        "qiskit_aer.primitives.Sampler does not support mid-circuit measurements when shots is None. "
        "Use circuit_knitting.utils.simulation.ExactSampler to generate exact distributions "
        "for each subexperiment.",
    )


def _terra_sampler_error() -> None:
    raise ValueError(
        "qiskit.primitives.Sampler does not support mid-circuit measurements. "
        "Use circuit_knitting.utils.simulation.ExactSampler to generate exact distributions "
        "for each subexperiment."
    )


def _bad_samplers_error() -> None:
    raise ValueError(
        "The samplers input argument must be either an instance of qiskit.primitives.BaseSampler "
        "or a mapping from partition labels to qiskit.primitives.BaseSampler instances."
    )
