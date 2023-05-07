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
from enum import Enum

import numpy as np
from qiskit.circuit import Gate
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

from .qpd_basis import QPDBasis
from .instructions import QPDMeasure
from ...utils.iteration import unique_by_id


class WeightType(Enum):
    """Type of weight."""

    #: A weight given in proportion to its exact weight
    EXACT = 1

    #: A weight that was determined through some sampling procedure
    SAMPLED = 2


def generate_qpd_samples(
    qpd_bases: Sequence[QPDBasis], num_samples: int = 1000
) -> dict[tuple[int, ...], tuple[int, WeightType]]:
    """
    Generate random quasiprobability decompositions.

    Args:
        qpd_bases: The :class:`QPDBasis` objects from which to sample
        num_samples: Number of random samples to generate
    Returns:
        (dict): A mapping from a given decomposition to its weight.
            Keys are tuples of indices -- one index per decomposition in the circuit. The indices
            correspond to a specific decomposition mapping which will be applied to each gate in
            the decomposition.
            Values are tuples.  The first element is a number corresponding to the
            weight of the contribution.  The second element is the :class:`WeightType`,
            either ``EXACT`` or ``SAMPLED``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if len(qpd_bases) == 0:
        # This case must be handled explicitly, as it is not handled correctly
        # by the `zip()` call below.
        return {(): (num_samples, WeightType.EXACT)}

    # Loop through each gate and sample from its distribution num_samples times
    samples_by_decomp = []
    for basis in qpd_bases:
        # All gates in a decomp should have same QPDBasis, so we sample from gate_0 object
        samples_by_decomp.append(
            np.random.choice(
                range(len(basis.probabilities)),
                num_samples,
                p=basis.probabilities,
            )
        )

    # Form the joint samples, collecting them into a dict with counts for each
    random_samples: dict[tuple[int, ...], int] = {}
    for decomp_ids in zip(*samples_by_decomp):
        # Increment the counter for this basis selection
        random_samples[decomp_ids] = random_samples.setdefault(decomp_ids, 0) + 1

    return {k: (v, WeightType.SAMPLED) for k, v in random_samples.items()}


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
        ValueError:
            Cannot decompose gate with unbound parameters
    """
    try:
        f = _qpdbasis_from_gate_funcs[gate.name]
    except KeyError:
        raise ValueError(f"Gate not supported: {gate.name}") from None
    else:
        return f(gate)


@_register_qpdbasis_from_gate("rxx", "ryy", "rzz", "crx", "cry", "crz")
def _(gate: RXXGate | RYYGate | RZZGate | CRXGate | CRYGate | CRZGate):
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
    except (TypeError) as err:
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

    coeffs = _generate_coefficients(theta)

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


def _generate_coefficients(theta: float):
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

    return coeffs
