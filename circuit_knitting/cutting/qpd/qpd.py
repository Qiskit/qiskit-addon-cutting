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
    ControlledGate,
    Instruction,
    ClassicalRegister,
    CircuitInstruction,
    Measure,
    Reset,
)
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates import (
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    TGate,
    TdgGate,
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    CSGate,
    CSdgGate,
    CSXGate,
    RXXGate,
    RYYGate,
    RZZGate,
    CRXGate,
    CRYGate,
    CRZGate,
    ECRGate,
    CPhaseGate,
    SwapGate,
    iSwapGate,
    DCXGate,
)
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.utils import deprecate_func

from .qpd_basis import QPDBasis
from .instructions import BaseQPDGate, TwoQubitQPDGate, QPDMeasure
from ..instructions import Move
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
    running_product: list[float],
    state: Sequence[int],
    coeff_probabilities: Sequence[npt.NDArray[np.float64]],
):
    """
    Update the ``running_product`` list after the ``state`` has been incremented.

    This snippet is used twice in
    :func:`_generate_exact_weights_and_conditional_probabilities_assume_sorted`;
    hence, we write it only once here.
    """
    try:
        prev = running_product[-2]
    except IndexError:
        prev = 1.0
    running_product[-1] = prev * coeff_probabilities[len(state) - 1][state[-1]]


def _generate_exact_weights_and_conditional_probabilities_assume_sorted(
    coeff_probabilities: Sequence[npt.NDArray[np.float64]], threshold: float
):
    r"""
    Determine all exact weights above ``threshold`` and the conditional probabilities necessary to sample efficiently from all other weights.

    Each yielded element will be a 2-tuple, the first element of which will be
    a ``state``, represented by a tuple of ``int``\ s.

    If ``state`` contains the same number of elements as
    ``coeff_probabilities``, then the second element will be a ``float``,
    greater than or equal to ``threshold``, that contains the exact probability
    of this ``state``.

    If, on the other hand, ``state`` contains fewer elements than
    ``coeff_probabilities``, then the second element is a 1D ``np.ndarray`` of
    the conditional probabilities that can be used to sample the remaining
    weights that have not been evaluated exactly (i.e., are below
    ``threshold``).  These will all be normalized *except* the
    top-level one, which is given by ``state == ()``.

    If there are no exact weights below a given level, then the conditional
    probabilities at that level are equivalent to the corresponding element of
    `coeff_probabilities``, and this generator does not yield them.  Skipping
    redundant conditional probabilities, like this, allows for efficient
    sampling without traversing the entire tree, which is desirable since the
    tree grows exponentially with the number of cut gates.  By contrast, the
    performance with the provided strategy is no worse than linear in
    ``threshold``, in both time and memory.

    This function assumes each element of ``coeff_probabilities`` contains
    non-negative numbers, ordered largest to smallest.  For a generalization
    that allows ``coeff_probabilities`` to be provided in any order, see
    :func:`_generate_exact_weights_and_conditional_probabilities`.
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
                # There were some exact weights found below us, so we need to
                # yield the conditional probabilities, unless all probabilities
                # at this level are zero *and* it's not the top level.
                current_condprobs = running_conditional_probabilities.pop()
                current_condprobs[
                    np.isclose(current_condprobs, 0, atol=_NONZERO_ATOL)
                ] = 0.0
                if not state:
                    # Don't normalize the top-level conditional probabilities.
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
    """
    Generate exact weights and conditional probabilities.

    This is identical in behavior to
    :func:`_generate_exact_weights_and_conditional_probabilities_assume_sorted`,
    except it makes no assumption on the order of the coefficients.
    """
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
            # In this case, `probability` is actually a vector of conditional
            # probabilities, so apply the inverse permutation.
            probability = probability[ipermutations[len(coeff_indices)]]
        yield orig_coeff_indices, probability


def generate_qpd_weights(
    qpd_bases: Sequence[QPDBasis], num_samples: float = 1000
) -> dict[tuple[int, ...], tuple[float, WeightType]]:
    """
    Generate weights from the joint quasiprobability distribution.

    Each weight whose absolute value is above a threshold of ``1 /
    num_samples`` will be evaluated exactly. The remaining weights -- those in
    the tail of the distribution -- will then be sampled from, resulting in at
    most ``num_samples`` unique weights.

    Args:
        qpd_bases: The :class:`QPDBasis` objects from which to generate weights
        num_samples: Controls the number of weights to generate

    Returns:
        A mapping from a given decomposition to its weight.
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
    num_samples: float,
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
            # As described in the docstring to
            # _generate_exact_weights_and_conditional_probabilities_assume_sorted,
            # there are two possible pieces of information that might be
            # yielded by the generator.  The following branch distinguishes
            # between them.
            if len(map_ids) == len(independent_probabilities):
                # The generator produced a state together with its exact probability.
                weight = probability * num_samples
                retval[map_ids] = (weight, WeightType.EXACT)
            else:
                # The generator produced a partial state along with a vector of
                # conditional probabilities.
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
    # Insert the samples into the dict we are about to return.
    for outcome, count in random_samples.items():
        assert outcome not in retval
        retval[outcome] = (count * single_sample_weight, WeightType.SAMPLED)

    return retval


def _populate_samples(
    random_samples: dict[tuple[int, ...], int],
    num_desired: int,
    independent_probabilities: Sequence,
    conditional_probabilities: dict[tuple[int, ...], npt.NDArray[np.float64]],
    running_state: tuple[int, ...] = (),
) -> None:
    """
    Generate random samples from the conditional probabilitity distributions.

    Items get populated into the ``random_samples`` dict, rather than returned.

    This function is designed to call itself recursively.  The number of
    elements in ``running_state`` will match the recursion depth.
    """
    if running_state not in conditional_probabilities:
        # Everything below us is sampled, so we can sample directly from the
        # remaining independent probability distributions.
        samples_by_decomp = []
        for probs in independent_probabilities[len(running_state) :]:
            samples_by_decomp.append(
                np.random.choice(range(len(probs)), num_desired, p=probs)
            )
        for outcome, count in Counter(zip(*samples_by_decomp)).items():
            assert (running_state + outcome) not in random_samples
            random_samples[running_state + outcome] = count
        return

    # There are some exact weight(s) below us, so we must consider the
    # conditional probabilities at the current level.
    probs = conditional_probabilities[running_state]
    current_outcomes = np.random.choice(range(len(probs)), num_desired, p=probs)
    for current_outcome, count in Counter(current_outcomes).items():
        outcome = running_state + (current_outcome,)
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
    *,
    inplace: bool = False,
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

    if not inplace:
        circuit = circuit.copy()  # pragma: no cover

    if map_ids is not None:
        if len(instruction_ids) != len(map_ids):
            raise ValueError(
                f"The number of map IDs ({len(map_ids)}) must equal the number of "
                f"decompositions in the circuit ({len(instruction_ids)})."
            )
        # If mapping is specified, set each gate's mapping
        for i, decomp_gate_ids in enumerate(instruction_ids):
            for gate_id in decomp_gate_ids:
                circuit.data[gate_id].operation.basis_id = map_ids[i]

    # Convert all instances of BaseQPDGate in the circuit to Qiskit instructions
    _decompose_qpd_instructions(circuit, instruction_ids)

    return circuit


_qpdbasis_from_instruction_funcs: dict[str, Callable[[Instruction], QPDBasis]] = {}


def _register_qpdbasis_from_instruction(*args):
    def g(f):
        for name in args:
            _qpdbasis_from_instruction_funcs[name] = f
        return f

    return g


@deprecate_func(
    since="0.3.0",
    package_name="circuit-knitting-toolbox",
    removal_timeline="no earlier than v0.4.0",
    additional_msg=(
        "This function has been renamed to "
        "``circuit_knitting.cutting.qpd.qpdbasis_from_instruction()``."
    ),
)
def qpdbasis_from_gate(gate: Instruction) -> QPDBasis:  # pragma: no cover
    """
    Generate a :class:`.QPDBasis` object, given a supported operation.

    All two-qubit gates which implement the :meth:`~qiskit.circuit.Gate.to_matrix` method are
    supported.  This should include the vast majority of gates with no unbound
    parameters, but there are some special cases (see, e.g., `qiskit issue #10396
    <https://github.com/Qiskit/qiskit-terra/issues/10396>`__).

    The :class:`.Move` operation, which can be used to specify a wire cut,
    is also supported.

    Returns:
        The newly-instantiated :class:`QPDBasis` object

    Raises:
        ValueError: Instruction not supported.
        ValueError: Cannot decompose instruction with unbound parameters.
        ValueError: ``to_matrix`` conversion of two-qubit gate failed.
    """
    return qpdbasis_from_instruction(gate)


def qpdbasis_from_instruction(gate: Instruction, /) -> QPDBasis:
    """
    Generate a :class:`.QPDBasis` object, given a supported operation.

    All two-qubit gates which implement the :meth:`~qiskit.circuit.Gate.to_matrix` method are
    supported.  This should include the vast majority of gates with no unbound
    parameters, but there are some special cases (see, e.g., `qiskit issue #10396
    <https://github.com/Qiskit/qiskit-terra/issues/10396>`__).

    The :class:`.Move` operation, which can be used to specify a wire cut,
    is also supported.

    Returns:
        The newly-instantiated :class:`QPDBasis` object

    Raises:
        ValueError: Instruction not supported.
        ValueError: Cannot decompose instruction with unbound parameters.
        ValueError: ``to_matrix`` conversion of two-qubit gate failed.
    """
    try:
        f = _qpdbasis_from_instruction_funcs[gate.name]
    except KeyError:
        pass
    else:
        return f(gate)

    if isinstance(gate, Gate) and gate.num_qubits == 2:
        try:
            mat = gate.to_matrix()
        except Exception as ex:
            raise ValueError(
                f"`to_matrix` conversion of two-qubit gate ({gate.name}) failed. "
                "Often, this can be caused by unbound parameters."
            ) from ex
        d = TwoQubitWeylDecomposition(mat)
        u = _u_from_thetavec([d.a, d.b, d.c])
        retval = _nonlocal_qpd_basis_from_u(u)
        for operations in unique_by_id(m[0] for m in retval.maps):
            operations.insert(0, UnitaryGate(d.K2r))
            operations.append(UnitaryGate(d.K1r))
        for operations in unique_by_id(m[1] for m in retval.maps):
            operations.insert(0, UnitaryGate(d.K2l))
            operations.append(UnitaryGate(d.K1l))
        return retval

    raise ValueError(f"Instruction not supported: {gate.name}")


def _explicitly_supported_instructions() -> set[str]:
    """
    Return a set of instruction names with explicit support for automatic decomposition.

    These instructions are *explicitly* supported by :func:`qpdbasis_from_instruction`.
    Other instructions may be supported too, via a KAK decomposition.

    Returns:
        Set of gate names supported for automatic decomposition.
    """
    return set(_qpdbasis_from_instruction_funcs)


def _copy_unique_sublists(lsts: tuple[list, ...], /) -> tuple[list, ...]:
    """
    Copy each list in a sequence of lists while preserving uniqueness.

    This is useful to ensure that the two sets of ``maps`` in a
    :class:`QPDBasis` will be independent of each other.  This enables one to
    subsequently edit the ``maps`` independently of each other (e.g., to apply
    single-qubit pre- or post-rotations.
    """
    copy_by_id: dict[int, list] = {}
    for lst in lsts:
        if id(lst) not in copy_by_id:
            copy_by_id[id(lst)] = lst.copy()
    return tuple(copy_by_id[id(lst)] for lst in lsts)


def _u_from_thetavec(
    theta: np.typing.NDArray[np.float64] | Sequence[float], /
) -> np.typing.NDArray[np.complex128]:
    r"""
    Exponentiate the non-local portion of a KAK decomposition.

    This implements Eq. (6) of https://arxiv.org/abs/2006.11174v2:

    .. math::

       \exp [ i ( \sum_\alpha^3 \theta_\alpha \, \sigma_\alpha \otimes \sigma_\alpha ) ]
       =
       \sum_{\alpha=0}^3 u_\alpha \, \sigma_\alpha \otimes \sigma_\alpha

    where each :math:`\theta_\alpha` is assumed to be real, and
    :math:`u_\alpha` is complex in general.
    """
    theta = np.asarray(theta)
    if theta.shape != (3,):
        raise ValueError(
            f"theta vector has wrong shape: {theta.shape} (1D vector of length 3 expected)"
        )
    # First, we note that if we choose the basis vectors II, XX, YY, and ZZ,
    # then the following matrix represents one application of the summation in
    # the exponential:
    #
    #   0   θx  θy  θz
    #   θx  0  -θz -θy
    #   θy -θz  0  -θx
    #   θz -θy -θx  0
    #
    # This matrix is symmetric and can be exponentiated by diagonalizing it.
    # Its eigendecomposition is given by:
    eigvals = np.array(
        [
            -np.sum(theta),
            -theta[0] + theta[1] + theta[2],
            -theta[1] + theta[2] + theta[0],
            -theta[2] + theta[0] + theta[1],
        ]
    )
    eigvecs = np.ones([1, 1]) / 2 - np.eye(4)
    # Finally, we exponentiate the eigenvalues of the matrix in diagonal form.
    # We also project to the vector [1,0,0,0] on the right, since the
    # multiplicative identity is given by II.
    return np.transpose(eigvecs) @ (np.exp(1j * eigvals) * eigvecs[:, 0])


def _nonlocal_qpd_basis_from_u(
    u: np.typing.NDArray[np.complex128] | Sequence[complex], /
) -> QPDBasis:
    u = np.asarray(u)
    if u.shape != (4,):
        raise ValueError(
            f"u vector has wrong shape: {u.shape} (1D vector of length 4 expected)"
        )
    # The following operations are described in Sec. 2.3 of
    # https://quantum-journal.org/papers/q-2021-01-28-388/
    #
    # Projective measurements in each basis
    A0x = [HGate(), QPDMeasure(), HGate()]
    A0y = [SXGate(), QPDMeasure(), SXdgGate()]
    A0z = [QPDMeasure()]
    # Single qubit rotations that swap two axes.  There are "plus" and "minus"
    # versions of these rotations.  The "minus" rotations also flip the sign
    # along that axis.
    Axyp = [SGate(), YGate()]
    Axym = [ZGate()] + Axyp
    Ayzp = [SXGate(), ZGate()]
    Ayzm = [XGate()] + Ayzp
    Azxp = [HGate()]
    Azxm = [YGate()] + Azxp
    # Single qubit rotations by ±pi/4 about each axis.
    B0xp = [SXGate()]
    B0xm = [SXdgGate()]
    B0yp = [RYGate(0.5 * np.pi)]
    B0ym = [RYGate(-0.5 * np.pi)]
    B0zp = [SGate()]
    B0zm = [SdgGate()]
    # Projective measurements, each followed by the proper flip.
    Bxy = A0z + [XGate()]
    Byz = A0x + [YGate()]
    Bzx = A0y + [ZGate()]
    # The following values occur repeatedly in the coefficients
    uu01 = u[0] * np.conj(u[1])
    uu02 = u[0] * np.conj(u[2])
    uu03 = u[0] * np.conj(u[3])
    uu12 = u[1] * np.conj(u[2])
    uu23 = u[2] * np.conj(u[3])
    uu31 = u[3] * np.conj(u[1])
    coeffs, maps1, maps2 = zip(
        # First line of Eq. (19) in
        # https://quantum-journal.org/papers/q-2021-01-28-388/
        (np.abs(u[0]) ** 2, [], []),  # Identity
        (np.abs(u[1]) ** 2, [XGate()], [XGate()]),
        (np.abs(u[2]) ** 2, [YGate()], [YGate()]),
        (np.abs(u[3]) ** 2, [ZGate()], [ZGate()]),
        # Second line
        (2 * np.real(uu01), A0x, A0x),
        (2 * np.real(uu02), A0y, A0y),
        (2 * np.real(uu03), A0z, A0z),
        (0.5 * np.real(uu12), Axyp, Axyp),
        (-0.5 * np.real(uu12), Axyp, Axym),
        (-0.5 * np.real(uu12), Axym, Axyp),
        (0.5 * np.real(uu12), Axym, Axym),
        (0.5 * np.real(uu23), Ayzp, Ayzp),
        (-0.5 * np.real(uu23), Ayzp, Ayzm),
        (-0.5 * np.real(uu23), Ayzm, Ayzp),
        (0.5 * np.real(uu23), Ayzm, Ayzm),
        (0.5 * np.real(uu31), Azxp, Azxp),
        (-0.5 * np.real(uu31), Azxp, Azxm),
        (-0.5 * np.real(uu31), Azxm, Azxp),
        (0.5 * np.real(uu31), Azxm, Azxm),
        (-0.5 * np.real(uu01), B0xp, B0xp),
        (0.5 * np.real(uu01), B0xp, B0xm),
        (0.5 * np.real(uu01), B0xm, B0xp),
        (-0.5 * np.real(uu01), B0xm, B0xm),
        (-0.5 * np.real(uu02), B0yp, B0yp),
        (0.5 * np.real(uu02), B0yp, B0ym),
        (0.5 * np.real(uu02), B0ym, B0yp),
        (-0.5 * np.real(uu02), B0ym, B0ym),
        (-0.5 * np.real(uu03), B0zp, B0zp),
        (0.5 * np.real(uu03), B0zp, B0zm),
        (0.5 * np.real(uu03), B0zm, B0zp),
        (-0.5 * np.real(uu03), B0zm, B0zm),
        (-2 * np.real(uu12), Bxy, Bxy),
        (-2 * np.real(uu23), Byz, Byz),
        (-2 * np.real(uu31), Bzx, Bzx),
        # Third line
        (np.imag(uu01), A0x, B0xp),
        (-np.imag(uu01), A0x, B0xm),
        (np.imag(uu01), B0xp, A0x),
        (-np.imag(uu01), B0xm, A0x),
        (np.imag(uu02), A0y, B0yp),
        (-np.imag(uu02), A0y, B0ym),
        (np.imag(uu02), B0yp, A0y),
        (-np.imag(uu02), B0ym, A0y),
        (np.imag(uu03), A0z, B0zp),
        (-np.imag(uu03), A0z, B0zm),
        (np.imag(uu03), B0zp, A0z),
        (-np.imag(uu03), B0zm, A0z),
        (np.imag(uu12), Axyp, Bxy),
        (-np.imag(uu12), Axym, Bxy),
        (np.imag(uu12), Bxy, Axyp),
        (-np.imag(uu12), Bxy, Axym),
        (np.imag(uu23), Ayzp, Byz),
        (-np.imag(uu23), Ayzm, Byz),
        (np.imag(uu23), Byz, Ayzp),
        (-np.imag(uu23), Byz, Ayzm),
        (np.imag(uu31), Azxp, Bzx),
        (-np.imag(uu31), Azxm, Bzx),
        (np.imag(uu31), Bzx, Azxp),
        (-np.imag(uu31), Bzx, Azxm),
    )
    maps = list(zip(maps1, _copy_unique_sublists(maps2)))
    return QPDBasis(maps, coeffs)


@_register_qpdbasis_from_instruction("swap")
def _(unused_gate: SwapGate):
    return _nonlocal_qpd_basis_from_u([(1 + 1j) / np.sqrt(8)] * 4)


@_register_qpdbasis_from_instruction("iswap")
def _(unused_gate: iSwapGate):
    return _nonlocal_qpd_basis_from_u([0.5, 0.5j, 0.5j, 0.5])


@_register_qpdbasis_from_instruction("dcx")
def _(unused_gate: DCXGate):
    retval = qpdbasis_from_instruction(iSwapGate())
    # Modify basis according to DCXGate definition in Qiskit circuit library
    # https://github.com/Qiskit/qiskit-terra/blob/e9f8b7c50968501e019d0cb426676ac606eb5a10/qiskit/circuit/library/standard_gates/equivalence_library.py#L938-L944
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, SdgGate())
        operations.insert(0, HGate())
    for operations in unique_by_id(m[1] for m in retval.maps):
        operations.insert(0, SdgGate())
        operations.append(HGate())
    return retval


@_register_qpdbasis_from_instruction("rxx", "ryy", "rzz", "crx", "cry", "crz")
def _(gate: RXXGate | RYYGate | RZZGate | CRXGate | CRYGate | CRZGate):
    # Constructing a virtual two-qubit gate by sampling single-qubit operations - Mitarai et al
    # https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc/pdf
    if gate.name in ("rxx", "crx"):
        pauli = XGate()
        r_plus = SXGate()
        # x basis measurement (and back again)
        measurement_0 = [HGate(), QPDMeasure(), HGate()]
    elif gate.name in ("ryy", "cry"):
        pauli = YGate()
        r_plus = RYGate(0.5 * np.pi)
        # y basis measurement (and back again)
        measurement_0 = [SXGate(), QPDMeasure(), SXdgGate()]
    else:
        assert gate.name in ("rzz", "crz")
        pauli = ZGate()
        r_plus = SGate()
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

    theta = _theta_from_instruction(gate)

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
        rot = {"x": RXGate, "y": RYGate, "z": RZGate}[gate.name[2]](-theta)
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


@_register_qpdbasis_from_instruction("cs", "csdg")
def _(gate: CSGate | CSdgGate):
    theta = np.pi / 2
    rot_gate = TGate()
    if gate.name == "csdg":
        theta *= -1
        rot_gate = rot_gate.inverse()
    retval = qpdbasis_from_instruction(CRZGate(theta))
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, rot_gate)
    return retval


@_register_qpdbasis_from_instruction("cp")
def _(gate: CPhaseGate):
    theta = _theta_from_instruction(gate)
    retval = qpdbasis_from_instruction(CRZGate(theta))
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, PhaseGate(theta / 2))
    return retval


@_register_qpdbasis_from_instruction("csx")
def _(unused_gate: CSXGate):
    retval = qpdbasis_from_instruction(CRXGate(np.pi / 2))
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, TGate())
    return retval


@_register_qpdbasis_from_instruction("csxdg")
def _(unused_gate: ControlledGate):
    retval = qpdbasis_from_instruction(CRXGate(-np.pi / 2))
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, TdgGate())
    return retval


@_register_qpdbasis_from_instruction("cx", "cy", "cz", "ch")
def _(gate: CXGate | CYGate | CZGate | CHGate):
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

    if gate.name != "cz":
        # Modify `maps` to sandwich the target operations inside of basis rotations
        for operations in unique_by_id(m[1] for m in maps):
            if operations:
                if gate.name in ("cx", "cy"):
                    operations.insert(0, HGate())
                    operations.append(HGate())
                    if gate.name == "cy":
                        operations.insert(0, SdgGate())
                        operations.append(SGate())
                elif gate.name == "ch":
                    operations.insert(0, RYGate(-np.pi / 4))
                    operations.append(RYGate(np.pi / 4))

    coeffs = [0.5, 0.5, 0.5, -0.5, 0.5, -0.5]

    return QPDBasis(maps, coeffs)


@_register_qpdbasis_from_instruction("ecr")
def _(unused_gate: ECRGate):
    retval = qpdbasis_from_instruction(CXGate())
    # Modify basis according to ECRGate definition in Qiskit circuit library
    # https://github.com/Qiskit/qiskit-terra/blob/d9763523d45a747fd882a7e79cc44c02b5058916/qiskit/circuit/library/standard_gates/equivalence_library.py#L656-L663
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, SGate())
        operations.append(XGate())
    for operations in unique_by_id(m[1] for m in retval.maps):
        operations.insert(0, SXGate())
    return retval


def _theta_from_instruction(gate: Gate, /) -> float:
    param_gates = {"rxx", "ryy", "rzz", "crx", "cry", "crz", "cp"}

    # Internal function should only be called for supported gates
    assert gate.name in param_gates

    # If theta is a bound ParameterExpression, convert to float, else raise error.
    try:
        theta = float(gate.params[0])
    except TypeError as err:
        raise ValueError(
            f"Cannot decompose ({gate.name}) instruction with unbound parameters."
        ) from err

    return theta


@_register_qpdbasis_from_instruction("move")
def _(unused_gate: Move):
    i_measurement = [Reset()]
    x_measurement = [HGate(), QPDMeasure(), Reset()]
    y_measurement = [SXGate(), QPDMeasure(), Reset()]
    z_measurement = [QPDMeasure(), Reset()]

    prep_0 = [Reset()]
    prep_1 = [Reset(), XGate()]
    prep_plus = [Reset(), HGate()]
    prep_minus = [Reset(), XGate(), HGate()]
    prep_iplus = [Reset(), SXdgGate()]
    prep_iminus = [Reset(), XGate(), SXdgGate()]

    # https://arxiv.org/abs/1904.00102v2 Eqs. (12)-(19)
    maps1, maps2, coeffs = zip(
        (i_measurement, prep_0, 0.5),
        (i_measurement, prep_1, 0.5),
        (x_measurement, prep_plus, 0.5),
        (x_measurement, prep_minus, -0.5),
        (y_measurement, prep_iplus, 0.5),
        (y_measurement, prep_iminus, -0.5),
        (z_measurement, prep_0, 0.5),
        (z_measurement, prep_1, -0.5),
    )
    maps = list(zip(maps1, maps2))
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

    qpdgate_ids_2q = sorted(qpdgate_ids_2q)
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
