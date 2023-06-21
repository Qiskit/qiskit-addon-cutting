# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for conducting quasiprobability decomposition."""

from __future__ import annotations

from collections.abc import Sequence, Callable
from collections import Counter
from enum import Enum
import itertools
import logging
import math

import numpy as np
import numpy.typing as npt
from qiskit.circuit import (
    QuantumCircuit,
    Gate,
    ClassicalRegister,
    CircuitInstruction,
    Measure,
)
from qiskit.circuit.library.standard_gates import (
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    RXGate,
    RYGate,
    RZGate,
    CXGate,
    CZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    CRXGate,
    CRYGate,
    CRZGate,
)
from qiskit.utils import deprecate_func

from .qpd_basis import QPDBasis
from .instructions import BaseQPDGate, TwoQubitQPDGate, QPDMeasure
from ...utils.iteration import unique_by_id, strict_zip


logger = logging.getLogger(__name__)


# math.sin(math.pi) is just above 1e-16, so at 1e-14, we are well above that.
# Numbers like this can often come up in the QPD coefficients.
_NONZERO_ATOL = 1e-14


class WeightType(Enum):
    """Type of weight associated with a QPD sample."""

    #: A weight given in proportion to its exact weight
    EXACT = 1

    #: A weight that was determined through some sampling procedure
    SAMPLED = 2


def _min_filter_nonzero(vals: npt.NDArray[np.float64], *, atol=_NONZERO_ATOL):
    return np.min(vals[np.logical_not(np.isclose(vals, 0, atol=atol))])


def __update_running_product_after_increment(
    running_product, state, coeff_probabilities
):
    # This snippet is used twice in
    # _generate_exact_weights_and_conditional_probabilities_assume_sorted;
    # hence, we write it only once here.
    try:
        prev = running_product[-2]
    except IndexError:
        prev = 1.0
    running_product[-1] = prev * coeff_probabilities[len(state) - 1][state[-1]]


def _generate_exact_weights_and_conditional_probabilities_assume_sorted(
    coeff_probabilities: Sequence[npt.NDArray[np.float64]], threshold: float
):
    """Yield each combination whose product is above some threshold.

    Also yields conditional probabilities that can be used to sample the
    remaining possibilities.  These will all be normalized _except_ the
    top-level one.

    This function assumes each element of `coeff_probabilities` contains non-negative
    numbers, ordered largest to smallest.

    """
    assert len(coeff_probabilities) > 0

    next_pop = False  # Stores whether the next move is a pop or not
    state = [0]
    running_product = [coeff_probabilities[0][0]]
    running_conditional_probabilities: list[npt.NDArray[np.float64]] = []
    while True:
        if next_pop:
            # Pop
            state.pop()
            if len(state) + 1 == len(running_conditional_probabilities):
                # There were some exact weights found below us, so we likely
                # need to yield the conditional probabilities.
                current_condprobs = running_conditional_probabilities.pop()
                current_condprobs[
                    np.isclose(current_condprobs, 0, atol=_NONZERO_ATOL)
                ] = 0.0
                if not state:
                    # Don't renormalize the top-level one.
                    yield (), current_condprobs
                else:
                    norm = np.sum(current_condprobs)
                    if norm != 0:
                        # Some, but not all, weights below us are exact.  If,
                        # instead, _all_ weights had been exact, we could have
                        # skipped this yield, as there is zero probability of
                        # reaching this partial state when sampling.
                        yield tuple(state), current_condprobs / norm
                    # Update the factor one level up from the popped one.
                    running_conditional_probabilities[-1][state[-1]] *= norm
            if not state:
                # We're all done
                return
            running_product.pop()
            # Increment the state counter.
            state[-1] += 1
            if state[-1] != len(coeff_probabilities[len(state) - 1]):
                # Increment successful (no overflow).  We don't need to pop again.
                next_pop = False
                __update_running_product_after_increment(
                    running_product, state, coeff_probabilities
                )
        else:
            if running_product[-1] < threshold:
                next_pop = True
            elif len(state) < len(coeff_probabilities):
                # Append 0 to work toward a "full" `state`
                running_product.append(
                    running_product[-1] * coeff_probabilities[len(state)][0]
                )
                state.append(0)
            else:
                # `state` is full.  Yield first.
                yield tuple(state), running_product[-1]
                # Since we found something exact, we need to update running_conditional_probabilities.
                while len(running_conditional_probabilities) < len(coeff_probabilities):
                    running_conditional_probabilities.append(
                        np.array(
                            coeff_probabilities[len(running_conditional_probabilities)],
                            dtype=float,
                        )
                    )
                # It's exact, so we want no probability of sampling it going forward.
                running_conditional_probabilities[-1][state[-1]] = 0
                # Increment the state counter.
                state[-1] += 1
                if state[-1] == len(coeff_probabilities[-1]):
                    # The state counter has overflowed, so our next move should be a
                    # pop.
                    next_pop = True
                else:
                    __update_running_product_after_increment(
                        running_product, state, coeff_probabilities
                    )


def _invert_permutation(p):
    # https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def _generate_exact_weights_and_conditional_probabilities(
    coeff_probabilities: Sequence[npt.NDArray[np.float64]], threshold: float
):
    # No assumption on the order or on the sign.
    permutations = [np.argsort(cp)[::-1] for cp in coeff_probabilities]
    sorted_coeff_probabilities = [
        cp[permutation] for cp, permutation in zip(coeff_probabilities, permutations)
    ]
    ipermutations = [_invert_permutation(p) for p in permutations]
    for (
        coeff_indices,
        probability,
    ) in _generate_exact_weights_and_conditional_probabilities_assume_sorted(
        sorted_coeff_probabilities, threshold
    ):
        orig_coeff_indices = tuple(
            perm[idx] for perm, idx in zip(permutations, coeff_indices)
        )
        if len(coeff_indices) != len(sorted_coeff_probabilities):
            probability = probability[ipermutations[len(coeff_indices)]]
        yield orig_coeff_indices, probability


@deprecate_func(
    since="0.3.0",
    package_name="circuit-knitting-toolbox",
    removal_timeline="no earlier than v0.4.0",
    additional_msg=(
        "This function has been renamed to "
        "``circuit_knitting.cutting.qpd.generate_qpd_weights()``."
    ),
)
def generate_qpd_samples(
    qpd_bases: Sequence[QPDBasis], num_samples: float = 1000
) -> dict[tuple[int, ...], tuple[float, WeightType]]:  # pragma: no cover
    """
    Generate random quasiprobability decompositions.

    Deprecated since CKT 0.3.0.  This function has been renamed to
    :func:`.generate_qpd_weights`.
    """
    return generate_qpd_weights(qpd_bases, num_samples)


def generate_qpd_weights(
    qpd_bases: Sequence[QPDBasis], num_samples: float = 1000
) -> dict[tuple[int, ...], tuple[float, WeightType]]:
    """
    Generate random quasiprobability decompositions.

    Args:
        qpd_bases: The :class:`QPDBasis` objects from which to sample
        num_samples: Number of random samples to generate

    Returns:
        A mapping from a given decomposition to its sampled weight.
        Keys are tuples of indices -- one index per input :class:`QPDBasis`. The indices
        correspond to a specific decomposition mapping in the basis.

        Values are tuples.  The first element is a number corresponding to the
        weight of the contribution.  The second element is the :class:`WeightType`,
        either ``EXACT`` or ``SAMPLED``.
    """
    independent_probabilities = [np.asarray(basis.probabilities) for basis in qpd_bases]
    # In Python 3.7 and higher, dicts are guaranteed to remember their
    # insertion order.  For user convenience, we sort by exact weights first,
    # then sampled weights.  Within each, values are sorted largest to
    # smallest.
    lst = sorted(
        _generate_qpd_weights(independent_probabilities, num_samples).items(),
        key=lambda x: ((v := x[1])[1].value, -v[0]),
    )
    return dict(lst)


def _generate_qpd_weights(
    independent_probabilities: Sequence[npt.NDArray[np.float64]],
    num_samples: float = 1000,
    *,
    _samples_multiplier: int = 1,
) -> dict[tuple[int, ...], tuple[float, WeightType]]:
    if not num_samples >= 1:
        raise ValueError("num_samples must be at least 1.")

    retval: dict[tuple[int, ...], tuple[float, WeightType]] = {}

    threshold = 1 / num_samples

    # Determine if the smallest *nonzero* probability is above the threshold implied by
    # num_samples.  If so, then we can evaluate all weights exactly.
    smallest_probability = np.prod(
        [_min_filter_nonzero(probs) for probs in independent_probabilities]
    )
    if smallest_probability >= threshold:
        # All weights exactly
        logger.info("All exact weights")
        multiplier = num_samples if math.isfinite(num_samples) else 1.0
        for map_ids in itertools.product(
            *[range(len(probs)) for probs in independent_probabilities]
        ):
            probability = np.prod(
                [
                    probs[i]
                    for i, probs in strict_zip(map_ids, independent_probabilities)
                ]
            )
            if probability < _NONZERO_ATOL:
                continue
            retval[map_ids] = (multiplier * probability, WeightType.EXACT)
        return retval

    conditional_probabilities: dict[tuple[int, ...], npt.NDArray[np.float64]] = {}
    weight_to_sample = 1.0

    largest_probability = np.prod(
        [np.max(probs) for probs in independent_probabilities]
    )
    if not largest_probability >= threshold:
        logger.info("No exact weights")
    else:
        logger.info("Some exact weights")
        for (
            map_ids,
            probability,
        ) in _generate_exact_weights_and_conditional_probabilities(
            independent_probabilities, threshold
        ):
            if len(map_ids) == len(independent_probabilities):
                weight = probability * num_samples
                retval[map_ids] = (weight, WeightType.EXACT)
            else:
                # Despite the variable name, `probability` is the sequence of
                # conditional probabilities, not a *single* probability
                conditional_probabilities[map_ids] = probability
                if map_ids == ():
                    weight_to_sample = np.sum(probability)
                    conditional_probabilities[map_ids] /= weight_to_sample

    # Loop through each gate and sample from the remainder of the distribution.

    # Start by rescaling.
    weight_to_sample *= num_samples
    # The following variable, `samples_needed`, must be integer and at least 1.
    # `_samples_multiplier` will typically be 1, but we set it higher in
    # testing to collect additional statistics, faster.
    samples_needed = math.ceil(weight_to_sample) * _samples_multiplier
    # At the time of writing, the below assert should never fail.  But if
    # future code changes result in inputs where it _may_ fail, then the only
    # thing that should be needed if this is reached is to return `retval` in
    # this case, since presumably it must contain all weights as exact weights.
    assert samples_needed >= 1
    single_sample_weight = weight_to_sample / samples_needed

    # Figure out if we've reached the special case where everything except
    # *one* weight has been calculated exactly, so there's only one thing left
    # to sample.  If that's the case, then we can calculate that one remaining
    # weight exactly as well and skip sampling.
    if conditional_probabilities:
        running_state: tuple[int, ...] = ()
        while len(running_state) < len(independent_probabilities):
            # If it's missing from `conditional_probabilities`, that just means
            # to use the corresponding entry in `independent_probabilities`.
            try:
                probs = conditional_probabilities[running_state]
            except KeyError:
                probs = independent_probabilities[len(running_state)]
            x = np.flatnonzero(probs)
            assert len(x) != 0
            if len(x) > 1:
                break
            running_state += (x[0],)
        else:
            assert running_state not in retval
            retval[running_state] = (weight_to_sample, WeightType.EXACT)
            return retval

    # Form the joint samples, collecting them into a dict with counts for each.
    random_samples: dict[tuple[int, ...], int] = {}
    _populate_samples(
        random_samples,
        samples_needed,
        independent_probabilities,
        conditional_probabilities,
    )
    # Insert the samples in the dict we are about to return.
    for outcome, count in random_samples.items():
        assert outcome not in retval
        retval[outcome] = (count * single_sample_weight, WeightType.SAMPLED)

    return retval


def _populate_samples(
    random_samples: dict[tuple[int, ...], int],
    num_desired: int,
    independent_probabilities: Sequence,
    conditional_probabilities,
    runner: tuple[int, ...] = (),
) -> None:
    if runner not in conditional_probabilities:
        # Everything below us is sampled, so we can sample directly from the
        # remaining independent probability distributions.
        samples_by_decomp = []
        for probs in independent_probabilities[len(runner) :]:
            samples_by_decomp.append(
                np.random.choice(range(len(probs)), num_desired, p=probs)
            )
        for outcome, count in Counter(zip(*samples_by_decomp)).items():
            assert (runner + outcome) not in random_samples
            random_samples[runner + outcome] = count
        return

    # There is some exact stuff below us, so we must consider the conditional
    # probabilities at the current level.
    probs = conditional_probabilities[runner]
    current_outcomes = np.random.choice(range(len(probs)), num_desired, p=probs)
    for current_outcome, count in Counter(current_outcomes).items():
        outcome = runner + (current_outcome,)
        if len(outcome) == len(independent_probabilities):
            # It's a full one
            assert outcome not in random_samples
            random_samples[outcome] = count
        else:
            # Recurse
            _populate_samples(
                random_samples,
                count,
                independent_probabilities,
                conditional_probabilities,
                outcome,
            )


def decompose_qpd_instructions(
    circuit: QuantumCircuit,
    instruction_ids: Sequence[Sequence[int]],
    map_ids: Sequence[int] | None = None,
) -> QuantumCircuit:
    r"""
    Replace all QPD instructions in the circuit with local Qiskit operations and measurements.

    Args:
        circuit: The circuit containing QPD instructions
        instruction_ids: A 2D sequence, such that each inner sequence corresponds to indices
            of instructions comprising one decomposition in the circuit. The elements within a
            common sequence belong to a common decomposition and should be sampled together.
        map_ids: Indices to a specific linear mapping to be applied to the decompositions
            in the circuit. If no map IDs are provided, the circuit will be decomposed randomly
            according to the decompositions' joint probability distribution.

    Returns:
        Circuit which has had all its :class:`BaseQPDGate` instances decomposed into local operations.

        The circuit will contain a new, final classical register to contain the QPD measurement
        outcomes (accessible at ``retval.cregs[-1]``).

    Raises:
        ValueError: An index in ``instruction_ids`` corresponds to a gate which is not a
            :class:`BaseQPDGate` instance.
        ValueError: A list within instruction_ids is not length 1 or 2.
        ValueError: The total number of indices in ``instruction_ids`` does not equal the number
            of :class:`BaseQPDGate` instances in the circuit.
        ValueError: Gates within the same decomposition hold different QPD bases.
        ValueError: Length of ``map_ids`` does not equal the number of decompositions in the circuit.
    """
    _validate_qpd_instructions(circuit, instruction_ids)
    new_qc = circuit.copy()

    if map_ids is not None:
        if len(instruction_ids) != len(map_ids):
            raise ValueError(
                f"The number of map IDs ({len(map_ids)}) must equal the number of "
                f"decompositions in the circuit ({len(instruction_ids)})."
            )
        # If mapping is specified, set each gate's mapping
        for i, decomp_gate_ids in enumerate(instruction_ids):
            for gate_id in decomp_gate_ids:
                new_qc.data[gate_id].operation.basis_id = map_ids[i]

    # Convert all instances of BaseQPDGate in the circuit to Qiskit instructions
    _decompose_qpd_instructions(new_qc, instruction_ids)

    return new_qc


_qpdbasis_from_gate_funcs: dict[str, Callable[[Gate], QPDBasis]] = {}


def _register_qpdbasis_from_gate(*args):
    def g(f):
        for name in args:
            _qpdbasis_from_gate_funcs[name] = f
        return f

    return g


def qpdbasis_from_gate(gate: Gate) -> QPDBasis:
    """
    Generate a QPDBasis object, given a supported operation.

    This method currently supports 8 operations:
        - :class:`~qiskit.circuit.library.RXXGate`
        - :class:`~qiskit.circuit.library.RYYGate`
        - :class:`~qiskit.circuit.library.RZZGate`
        - :class:`~qiskit.circuit.library.CRXGate`
        - :class:`~qiskit.circuit.library.CRYGate`
        - :class:`~qiskit.circuit.library.CRZGate`
        - :class:`~qiskit.circuit.library.CXGate`
        - :class:`~qiskit.circuit.library.CZGate`

    Returns:
        The newly-instantiated :class:`QPDBasis` object

    Raises:
        ValueError: Cannot decompose gate with unbound parameters.
    """
    try:
        f = _qpdbasis_from_gate_funcs[gate.name]
    except KeyError:
        raise ValueError(f"Gate not supported: {gate.name}") from None
    else:
        return f(gate)


@_register_qpdbasis_from_gate("rxx", "ryy", "rzz", "crx", "cry", "crz")
def _(gate: RXXGate | RYYGate | RZZGate | CRXGate | CRYGate | CRZGate):
    # Constructing a virtual two-qubit gate by sampling single-qubit operations - Mitarai et al
    # https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc/pdf
    if gate.name in ("rxx", "crx"):
        pauli = XGate()
        r_plus = RXGate(0.5 * np.pi)
        # x basis measurement (and back again)
        measurement_0 = [HGate(), QPDMeasure(), HGate()]
    elif gate.name in ("ryy", "cry"):
        pauli = YGate()
        r_plus = RYGate(0.5 * np.pi)
        # y basis measurement (and back again)
        measurement_0 = [SdgGate(), HGate(), QPDMeasure(), HGate(), SGate()]
    else:
        assert gate.name in ("rzz", "crz")
        pauli = ZGate()
        r_plus = RZGate(0.5 * np.pi)
        # z basis measurement
        measurement_0 = [QPDMeasure()]

    r_minus = r_plus.inverse()
    measurement_1 = measurement_0.copy()

    # Specify the operations to be inserted for systems 1 and 2
    maps = [
        ([], []),  # Identity
        ([pauli], [pauli]),
        (measurement_0, [r_plus]),
        (measurement_0, [r_minus]),
        ([r_plus], measurement_1),
        ([r_minus], measurement_1),
    ]

    # If theta is a bound ParameterExpression, convert to float, else raise error.
    try:
        theta = float(gate.params[0])
    except TypeError as err:
        raise ValueError(
            f"Cannot decompose ({gate.name}) gate with unbound parameters."
        ) from err

    if gate.name[0] == "c":
        # Following Eq. (C.4) of https://arxiv.org/abs/2205.00016v2,
        # which implements controlled single qubit rotations in terms of a
        # parametric two-qubit rotation.
        theta = -theta / 2
        # Modify `maps`
        for operations in unique_by_id(m[0] for m in maps):
            if operations and gate.name != "crz":
                if gate.name == "cry":
                    operations.insert(0, SGate())
                    operations.append(SdgGate())
                operations.insert(0, HGate())
                operations.append(HGate())
        rot = type(r_plus)(-theta)
        for operations in unique_by_id(m[1] for m in maps):
            operations.append(rot)

    # Generate QPD coefficients for gates specified by Mitarai et al -
    # https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc/pd
    theta_prime = -theta / 2  # theta as defined by Mitarai and Fujii
    cs_theta_prime = np.cos(theta_prime) * np.sin(theta_prime)
    coeffs = [
        np.cos(theta_prime) ** 2,
        np.sin(theta_prime) ** 2,
        # Qiskit defines RZ(theta) = exp(-i*Z*theta/2), i.e., with a minus
        # sign in the exponential; hence r_plus really represents
        # exp(-i*Z*pi/4), and similar for RX and RY.  Following Mitarai and
        # Fujii Eq. (6), the correct signs are therefore given by
        -cs_theta_prime,
        cs_theta_prime,
        -cs_theta_prime,
        cs_theta_prime,
    ]

    return QPDBasis(maps, coeffs)


@_register_qpdbasis_from_gate("cz", "cx")
def _(gate: CZGate | CXGate):
    # Constructing a virtual two-qubit gate by sampling single-qubit operations - Mitarai et al
    # https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc/pdf
    measurement_0 = [SdgGate(), QPDMeasure()]
    measurement_1 = measurement_0.copy()

    # Specify the operations to be inserted for systems 1 and 2
    maps = [
        ([SdgGate()], [SdgGate()]),
        ([SGate()], [SGate()]),
        # R-R+ simplifies to I
        (measurement_0, []),
        (measurement_0, [ZGate()]),
        # R-R+ simplifies to I
        ([], measurement_1),
        ([ZGate()], measurement_1),
    ]

    if gate.name == "cx":
        # Modify `maps` to sandwich the target operations inside of Hadamards
        for operations in {id(m[1]): m[1] for m in maps}.values():
            if operations:
                operations.insert(0, HGate())
                operations.append(HGate())

    coeffs = [0.5, 0.5, 0.5, -0.5, 0.5, -0.5]

    return QPDBasis(maps, coeffs)


def _validate_qpd_instructions(
    circuit: QuantumCircuit, instruction_ids: Sequence[Sequence[int]]
):
    """Ensure the indices in instruction_ids correctly describe all the decompositions in the circuit."""
    # Make sure all instruction_ids correspond to QPDGates, and make sure each QPDGate in a given decomposition has
    # an equivalent QPDBasis to its sibling QPDGates
    for decomp_ids in instruction_ids:
        if len(decomp_ids) not in [1, 2]:
            raise ValueError(
                "Each decomposition must contain either one or two elements. Found a "
                f"decomposition with ({len(decomp_ids)}) elements."
            )
        if not isinstance(circuit.data[decomp_ids[0]].operation, BaseQPDGate):
            raise ValueError(
                f"A circuit data index ({decomp_ids[0]}) corresponds to a non-QPDGate "
                f"({circuit.data[decomp_ids[0]].operation.name})."
            )
        compare_basis = circuit.data[decomp_ids[0]].operation.basis
        for gate_id in decomp_ids:
            if not isinstance(circuit.data[gate_id].operation, BaseQPDGate):
                raise ValueError(
                    f"A circuit data index ({gate_id}) corresponds to a non-QPDGate "
                    f"({circuit.data[gate_id].operation.name})."
                )
            tmp_basis = circuit.data[gate_id].operation.basis
            if compare_basis != tmp_basis:
                raise ValueError(
                    "Gates within the same decomposition must share an equivalent QPDBasis."
                )

    # Make sure the total number of QPD gate indices equals the number of QPDGates in the circuit
    num_qpd_gates = sum(len(x) for x in instruction_ids)
    qpd_gate_total = 0
    for inst in circuit.data:
        if isinstance(inst.operation, BaseQPDGate):
            qpd_gate_total += 1
    if qpd_gate_total != num_qpd_gates:
        raise ValueError(
            f"The total number of QPDGates specified in instruction_ids ({num_qpd_gates}) "
            f"does not equal the number of QPDGates in the circuit ({qpd_gate_total})."
        )


def _decompose_qpd_measurements(
    circuit: QuantumCircuit, inplace: bool = True
) -> QuantumCircuit:
    """
    Create mid-circuit measurements.

    Convert all QPDMeasure instances to Measure instructions. Add any newly created
    classical bits to a new "qpd_measurements" register.
    """
    if not inplace:
        circuit = circuit.copy()  # pragma: no cover

    # Loop through the decomposed circuit to find QPDMeasure markers so we can
    # replace them with measurement instructions.  We can't use `_ids`
    # here because it refers to old indices, before the decomposition.
    qpd_measure_ids = [
        i
        for i, instruction in enumerate(circuit.data)
        if instruction.operation.name.lower() == "qpd_measure"
    ]

    # Create a classical register for the qpd measurement results.  This is
    # partly for convenience, partly to work around
    # https://github.com/Qiskit/qiskit-aer/issues/1660.
    reg = ClassicalRegister(len(qpd_measure_ids), name="qpd_measurements")
    circuit.add_register(reg)

    # Place the measurement instructions
    for idx, i in enumerate(qpd_measure_ids):
        gate = circuit.data[i]
        inst = CircuitInstruction(
            operation=Measure(), qubits=[gate.qubits], clbits=[reg[idx]]
        )
        circuit.data[i] = inst

    # If the user wants to access the qpd register, it will be the final
    # classical register of the returned circuit.
    assert circuit.cregs[-1] is reg

    return circuit


def _decompose_qpd_instructions(
    circuit: QuantumCircuit,
    instruction_ids: Sequence[Sequence[int]],
    inplace: bool = True,
) -> QuantumCircuit:
    """Decompose all BaseQPDGate instances, ignoring QPDMeasure()."""
    if not inplace:
        circuit = circuit.copy()  # pragma: no cover

    # Decompose any 2q QPDGates into single qubit QPDGates
    qpdgate_ids_2q = []
    for decomp in instruction_ids:
        if len(decomp) != 1:
            continue  # pragma: no cover
        if isinstance(circuit.data[decomp[0]].operation, TwoQubitQPDGate):
            qpdgate_ids_2q.append(decomp[0])
    data_id_offset = 0
    for i in qpdgate_ids_2q:
        inst = circuit.data[i + data_id_offset]
        qpdcirc_2q_decomp = inst.operation.definition
        inst1 = CircuitInstruction(
            qpdcirc_2q_decomp.data[0].operation, qubits=[inst.qubits[0]]
        )
        inst2 = CircuitInstruction(
            qpdcirc_2q_decomp.data[1].operation, qubits=[inst.qubits[1]]
        )
        circuit.data[i + data_id_offset] = inst1
        data_id_offset += 1
        circuit.data.insert(i + data_id_offset, inst2)

    # Decompose all the QPDGates (should all be single qubit now) into Qiskit operations
    new_instruction_ids = []
    for i, inst in enumerate(circuit.data):
        if isinstance(inst.operation, BaseQPDGate):
            new_instruction_ids.append(i)
    data_id_offset = 0
    for i in new_instruction_ids:
        inst = circuit.data[i + data_id_offset]
        qubits = inst.qubits
        # All gates in decomposition should be local
        assert len(qubits) == 1
        # Gather instructions with which we will replace the QPDGate
        tmp_data = []
        for data in inst.operation.definition.data:
            # Can ignore clbits here, as QPDGates don't use clbits directly
            assert data.clbits == ()
            tmp_data.append(CircuitInstruction(data.operation, qubits=[qubits[0]]))
        # Replace QPDGate with local operations
        if tmp_data:
            # Overwrite the QPDGate with first instruction
            circuit.data[i + data_id_offset] = tmp_data[0]
            # Append remaining instructions immediately after original QPDGate position
            for data in tmp_data[1:]:
                data_id_offset += 1
                circuit.data.insert(i + data_id_offset, data)

        # If QPDGate decomposes to an identity operation, just delete it
        else:
            del circuit.data[i + data_id_offset]
            data_id_offset -= 1

    _decompose_qpd_measurements(circuit)

    return circuit
