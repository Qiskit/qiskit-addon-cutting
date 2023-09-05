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
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import PauliList
from qiskit.primitives import BaseSampler, Sampler as TerraSampler, SamplerResult
from qiskit_aer.primitives import Sampler as AerSampler

from ..utils.observable_grouping import CommutingObservableGroup, ObservableCollection
from ..utils.iteration import strict_zip
from .qpd import (
    QPDBasis,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    generate_qpd_weights,
    decompose_qpd_instructions,
    WeightType,
)
from .cutting_decomposition import decompose_observables


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
        - A list of :class:`~qiskit.primitives.SamplerResult` instances -- one for each partition.
        - Coefficients corresponding to each unique subexperiment's sampling frequency

    Raises:
        ValueError: The number of requested samples must be at least one.
        ValueError: The types of ``circuits`` and ``subobservables`` arguments are incompatible.
        ValueError: ``SingleQubitQPDGate``\ s are not supported in unseparable circuits.
        ValueError: The keys for the input dictionaries are not equivalent.
        ValueError: One or more input circuit contains classical registers.
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
    subexperiments, coefficients = _generate_cutting_experiments(
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

    # Make sure all input circuits are clear of classical regs.
    # Submit a job for each circuit partition.
    jobs = {}
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
        jobs[label] = samplers_dict[label].run(subexperiments_dict[label])

    # Collect the results from each job, and add the number of qpd bits for each circuit to the metadata.
    results = {
        label: jobs[label].result() for label in sorted(subexperiments_dict.keys())
    }
    for label, result in results.items():
        for i, metadata in enumerate(result.metadata):
            metadata["num_qpd_bits"] = len(subexperiments_dict[label][i].cregs[0])

    # If the input was a circuit, the output results should be a single SamplerResult instance
    results_out = results
    if isinstance(circuits, QuantumCircuit):
        assert len(results_out.keys()) == 1
        results_out = results[list(results.keys())[0]]

    return CuttingExperimentResults(results=results_out, coeffs=coefficients)


def _append_measurement_circuit(
    qc: QuantumCircuit,
    cog: CommutingObservableGroup,
    /,
    *,
    qubit_locations: Sequence[int] | None = None,
    inplace: bool = False,
) -> QuantumCircuit:
    """Append a new classical register and measurement instructions for the given ``CommutingObservableGroup``.

    The new register will be named ``"observable_measurements"`` and will be
    the final register in the returned circuit, i.e. ``retval.cregs[-1]``.

    Args:
        qc: The quantum circuit
        cog: The commuting observable set for
            which to construct measurements
        qubit_locations: A ``Sequence`` whose length is the number of qubits
            in the observables, where each element holds that qubit's corresponding
            index in the circuit.  By default, the circuit and observables are assumed
            to have the same number of qubits, and the identity map
            (i.e., ``range(qc.num_qubits)``) is used.
        inplace: Whether to operate on the circuit in place (default: ``False``)

    Returns:
        The modified circuit
    """
    if qubit_locations is None:
        # By default, the identity map.
        if qc.num_qubits != cog.general_observable.num_qubits:
            raise ValueError(
                f"Quantum circuit qubit count ({qc.num_qubits}) does not match qubit "
                f"count of observable(s) ({cog.general_observable.num_qubits}).  "
                f"Try providing `qubit_locations` explicitly."
            )
        qubit_locations = range(cog.general_observable.num_qubits)
    else:
        if len(qubit_locations) != cog.general_observable.num_qubits:
            raise ValueError(
                f"qubit_locations has {len(qubit_locations)} element(s) but the "
                f"observable(s) have {cog.general_observable.num_qubits} qubit(s)."
            )
    if not inplace:
        qc = qc.copy()

    # Append the appropriate measurements to qc
    obs_creg = ClassicalRegister(len(cog.pauli_indices), name="observable_measurements")
    qc.add_register(obs_creg)
    # Implement the necessary basis rotations and measurements, as
    # in BackendEstimator._measurement_circuit().
    genobs_x = cog.general_observable.x
    genobs_z = cog.general_observable.z
    for clbit, subqubit in enumerate(cog.pauli_indices):
        # subqubit is the index of the qubit in the subsystem.
        # actual_qubit is its index in the system of interest (if different).
        actual_qubit = qubit_locations[subqubit]
        if genobs_x[subqubit]:
            if genobs_z[subqubit]:
                qc.sdg(actual_qubit)
            qc.h(actual_qubit)
        qc.measure(actual_qubit, obs_creg[clbit])

    return qc


def _get_mapping_ids_by_partition(
    circuits: Sequence[QuantumCircuit],
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Get indices to the QPD gates in each subcircuit and relevant map ids."""
    # Collect QPDGate id's and relevant map id's for each subcircuit
    subcirc_qpd_gate_ids: list[list[list[int]]] = []
    subcirc_map_ids: list[list[int]] = []
    decomp_ids = set()
    for circ in circuits:
        subcirc_qpd_gate_ids.append([])
        subcirc_map_ids.append([])
        for i, inst in enumerate(circ.data):
            if isinstance(inst.operation, SingleQubitQPDGate):
                try:
                    decomp_id = int(inst.operation.label.split("_")[-1])
                except (AttributeError, ValueError):
                    raise ValueError(
                        "SingleQubitQPDGate instances in input circuit(s) must have their "
                        'labels suffixed with "_<id>", where <id> is the index of the cut '
                        "relative to the other cuts in the circuit. For example, all "
                        "SingleQubitQPDGates belonging to the same cut, N, should have labels "
                        ' formatted as "<your_label>_N". This allows SingleQubitQPDGates '
                        "belonging to the same cut to be sampled jointly."
                    )
                decomp_ids.add(decomp_id)
                subcirc_qpd_gate_ids[-1].append([i])
                subcirc_map_ids[-1].append(decomp_id)

    return subcirc_qpd_gate_ids, subcirc_map_ids


def _get_bases_by_partition(
    circuits: Sequence[QuantumCircuit], subcirc_qpd_gate_ids: list[list[list[int]]]
) -> list[QPDBasis]:
    """Get a list of each unique QPD basis across the subcircuits."""
    # Collect the bases corresponding to each decomposed operation
    bases_dict = {}
    for i, subcirc in enumerate(subcirc_qpd_gate_ids):
        for basis_id in subcirc:
            decomp_id = int(
                circuits[i].data[basis_id[0]].operation.label.split("_")[-1]
            )
            bases_dict[decomp_id] = circuits[i].data[basis_id[0]].operation.basis
    bases = [bases_dict[key] for key in sorted(bases_dict.keys())]

    return bases


def _get_bases(circuit: QuantumCircuit) -> tuple[list[QPDBasis], list[list[int]]]:
    """Get a list of each unique QPD basis in the circuit and the QPDGate indices."""
    bases = []
    qpd_gate_ids = []
    for i, inst in enumerate(circuit):
        if isinstance(inst.operation, SingleQubitQPDGate):
            raise ValueError(
                "SingleQubitQPDGates are not supported in unseparable circuits."
            )
        if isinstance(inst.operation, TwoQubitQPDGate):
            bases.append(inst.operation.basis)
            qpd_gate_ids.append([i])

    return bases, qpd_gate_ids


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
