# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" pauli common functions """

import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.qasm import pi
from qiskit.quantum_info import Pauli  # pylint: disable=unused-import

logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def pauli_measurement(circuit, pauli, qreg, creg, barrier=False):
    """
    Add the proper post-rotation gate on the circuit.

    Args:
        circuit (QuantumCircuit): the circuit to be modified.
        pauli (Pauli): the pauli will be added.
        qreg (QuantumRegister): the quantum register associated with the circuit.
        creg (ClassicalRegister): the classical register associated with the circuit.
        barrier (bool, optional): whether or not add barrier before measurement.

    Returns:
        QuantumCircuit: the original circuit object with post-rotation gate
    """
    num_qubits = pauli.num_qubits
    for qubit_idx in range(num_qubits):
        if pauli.x[qubit_idx]:
            if pauli.z[qubit_idx]:
                # Measure Y
                circuit.sdg(qreg[qubit_idx])  # sdg
                circuit.h(qreg[qubit_idx])  # h
            else:
                # Measure X
                circuit.h(qreg[qubit_idx])  # h
        if barrier:
            circuit.barrier(qreg[qubit_idx])
        circuit.measure(qreg[qubit_idx], creg[qubit_idx])

    return circuit


def measure_pauli_z(data, pauli):
    """
    Appropriate post-rotations on the state are assumed.

    Args:
        data (dict): a dictionary of the form data = {'00000': 10} ({str: int})
        pauli (Pauli): a Pauli object

    Returns:
        float: Expected value of paulis given data
    """
    observable = 0.0
    num_shots = sum(data.values())
    p_z_or_x = np.logical_or(pauli.z, pauli.x)
    for key, value in data.items():
        bitstr = np.asarray(list(key))[::-1].astype(int).astype(bool)
        # pylint: disable=no-member
        sign = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p_z_or_x)) else 1.0
        observable += sign * value
    observable /= num_shots
    return observable


def covariance(data, pauli_1, pauli_2, avg_1, avg_2):
    """
    Compute the covariance matrix element between two
    Paulis, given the measurement outcome.
    Appropriate post-rotations on the state are assumed.

    Args:
        data (dict): a dictionary of the form data = {'00000': 10} ({str:int})
        pauli_1 (Pauli): a Pauli class member
        pauli_2 (Pauli): a Pauli class member
        avg_1 (float): expectation value of pauli_1 on `data`
        avg_2 (float): expectation value of pauli_2 on `data`

    Returns:
        float: the element of the covariance matrix between two Paulis
    """
    cov = 0.0
    num_shots = sum(data.values())

    if num_shots == 1:
        return cov

    p1_z_or_x = np.logical_or(pauli_1.z, pauli_1.x)
    p2_z_or_x = np.logical_or(pauli_2.z, pauli_2.x)
    for key, value in data.items():
        bitstr = np.asarray(list(key))[::-1].astype(int).astype(bool)
        # pylint: disable=no-member
        sign_1 = (
            -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p1_z_or_x)) else 1.0
        )
        sign_2 = (
            -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p2_z_or_x)) else 1.0
        )
        cov += (sign_1 - avg_1) * (sign_2 - avg_2) * value
    cov /= num_shots - 1
    return cov


# pylint: disable=invalid-name
def suzuki_expansion_slice_pauli_list(pauli_list, lam_coef, expansion_order):
    """
    Compute the list of pauli terms for a single slice of the suzuki expansion following the paper
    https://arxiv.org/pdf/quant-ph/0508139.pdf.

    Args:
        pauli_list (list[list[complex, Pauli]]): The slice's weighted Pauli list for the
                                                 suzuki expansion
        lam_coef (float): The parameter lambda as defined in said paper,
                          adjusted for the evolution time and the number of time slices
        expansion_order (int): The order for suzuki expansion

    Returns:
        list: slice pauli list
    """
    if expansion_order == 1:
        half = [[lam_coef / 2 * c, p] for c, p in pauli_list]
        res = half + list(reversed(half))
    else:
        p_k = (4 - 4 ** (1 / (2 * expansion_order - 1))) ** -1
        side_base = suzuki_expansion_slice_pauli_list(
            pauli_list, lam_coef * p_k, expansion_order - 1
        )
        side = side_base * 2
        middle = suzuki_expansion_slice_pauli_list(
            pauli_list, lam_coef * (1 - 4 * p_k), expansion_order - 1
        )
        res = side + middle + side
    return res


def check_commutativity(op_1, op_2, anti=False):
    """
    Check the (anti-)commutativity between two operators.

    Args:
        op_1 (WeightedPauliOperator): operator
        op_2 (WeightedPauliOperator): operator
        anti (bool): if True, check anti-commutativity, otherwise check commutativity.

    Returns:
        bool: whether or not two operators are commuted or anti-commuted.
    """
    com = op_1 * op_2 - op_2 * op_1 if not anti else op_1 * op_2 + op_2 * op_1
    com.simplify()
    return bool(com.is_empty())


# pylint: disable=too-many-statements
def evolution_instruction(
    pauli_list,
    evo_time,
    num_time_slices,
    controlled=False,
    power=1,
    use_basis_gates=True,
    shallow_slicing=False,
    barrier=False,
):
    """
    Construct the evolution circuit according to the supplied specification.

    Args:
        pauli_list (list([[complex, Pauli]])): The list of pauli terms corresponding
                                               to a single time slice to be evolved
        evo_time (Union(complex, float, Parameter, ParameterExpression)): The evolution time
        num_time_slices (int): The number of time slices for the expansion
        controlled (bool, optional): Controlled circuit or not
        power (int, optional): The power to which the unitary operator is to be raised
        use_basis_gates (bool, optional): boolean flag for indicating only using basis
                                          gates when building circuit.
        shallow_slicing (bool, optional): boolean flag for indicating using shallow
                                          qc.data reference repetition for slicing
        barrier (bool, optional): whether or not add barrier for every slice

    Returns:
        Instruction: The Instruction corresponding to specified evolution.

    Raises:
        AlgorithmError: power must be an integer and greater or equal to 1
        ValueError: Unrecognized pauli
    """

    if not isinstance(power, int) or power < 1:
        raise AlgorithmError("power must be an integer and greater or equal to 1.")

    state_registers = QuantumRegister(pauli_list[0][1].num_qubits)
    if controlled:
        inst_name = f"Controlled-Evolution^{power}"
        ancillary_registers = QuantumRegister(1)
        qc_slice = QuantumCircuit(state_registers, ancillary_registers, name=inst_name)
    else:
        inst_name = f"Evolution^{power}"
        qc_slice = QuantumCircuit(state_registers, name=inst_name)

    # for each pauli [IXYZ]+, record the list of qubit pairs needing CX's
    cnot_qubit_pairs = [None] * len(pauli_list)
    # for each pauli [IXYZ]+, record the highest index of the nontrivial pauli gate (X,Y, or Z)
    top_xyz_pauli_indices = [-1] * len(pauli_list)

    for pauli_idx, pauli in enumerate(reversed(pauli_list)):
        n_qubits = pauli[1].num_qubits
        # changes bases if necessary
        nontrivial_pauli_indices = []
        for qubit_idx in range(n_qubits):
            # pauli I
            if not pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
                continue

            if cnot_qubit_pairs[pauli_idx] is None:
                nontrivial_pauli_indices.append(qubit_idx)

            if pauli[1].x[qubit_idx]:
                # pauli X
                if not pauli[1].z[qubit_idx]:
                    if use_basis_gates:
                        qc_slice.h(state_registers[qubit_idx])
                    else:
                        qc_slice.h(state_registers[qubit_idx])
                # pauli Y
                elif pauli[1].z[qubit_idx]:
                    if use_basis_gates:
                        qc_slice.u(pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                    else:
                        qc_slice.rx(pi / 2, state_registers[qubit_idx])
            # pauli Z
            elif pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
                pass
            else:
                raise ValueError(f"Unrecognized pauli: {pauli[1]}")

        if nontrivial_pauli_indices:
            top_xyz_pauli_indices[pauli_idx] = nontrivial_pauli_indices[-1]

        # insert lhs cnot gates
        if cnot_qubit_pairs[pauli_idx] is None:
            cnot_qubit_pairs[pauli_idx] = list(
                zip(
                    sorted(nontrivial_pauli_indices)[:-1],
                    sorted(nontrivial_pauli_indices)[1:],
                )
            )

        for pair in cnot_qubit_pairs[pauli_idx]:
            qc_slice.cx(state_registers[pair[0]], state_registers[pair[1]])

        # insert Rz gate
        if top_xyz_pauli_indices[pauli_idx] >= 0:

            # Because Parameter does not support complexity number operation; thus, we do
            # the following tricks to generate parameterized instruction.
            # We assume the coefficient in the pauli is always real. and can not do imaginary time
            # evolution
            if isinstance(evo_time, (Parameter, ParameterExpression)):
                lam = 2.0 * pauli[0] / num_time_slices
                lam = lam.real if lam.imag == 0 else lam
                lam = lam * evo_time
            else:
                lam = (2.0 * pauli[0] * evo_time / num_time_slices).real

            if not controlled:
                if use_basis_gates:
                    qc_slice.p(lam, state_registers[top_xyz_pauli_indices[pauli_idx]])
                else:
                    qc_slice.rz(lam, state_registers[top_xyz_pauli_indices[pauli_idx]])
            else:
                if use_basis_gates:
                    qc_slice.p(
                        lam / 2, state_registers[top_xyz_pauli_indices[pauli_idx]]
                    )
                    qc_slice.cx(
                        ancillary_registers[0],
                        state_registers[top_xyz_pauli_indices[pauli_idx]],
                    )
                    qc_slice.p(
                        -lam / 2, state_registers[top_xyz_pauli_indices[pauli_idx]]
                    )
                    qc_slice.cx(
                        ancillary_registers[0],
                        state_registers[top_xyz_pauli_indices[pauli_idx]],
                    )
                else:
                    qc_slice.crz(
                        lam,
                        ancillary_registers[0],
                        state_registers[top_xyz_pauli_indices[pauli_idx]],
                    )

        # insert rhs cnot gates
        for pair in reversed(cnot_qubit_pairs[pauli_idx]):
            qc_slice.cx(state_registers[pair[0]], state_registers[pair[1]])

        # revert bases if necessary
        for qubit_idx in range(n_qubits):
            if pauli[1].x[qubit_idx]:
                # pauli X
                if not pauli[1].z[qubit_idx]:
                    if use_basis_gates:
                        qc_slice.h(state_registers[qubit_idx])
                    else:
                        qc_slice.h(state_registers[qubit_idx])
                # pauli Y
                elif pauli[1].z[qubit_idx]:
                    if use_basis_gates:
                        qc_slice.u(-pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                    else:
                        qc_slice.rx(-pi / 2, state_registers[qubit_idx])
    # repeat the slice
    if shallow_slicing:
        logger.info(
            "Under shallow slicing mode, the qc.data reference is repeated shallowly. "
            "Thus, changing gates of one slice of the output circuit might affect "
            "other slices."
        )
        if barrier:
            qc_slice.barrier(state_registers)
        qc_slice.data *= num_time_slices * power
        qc = qc_slice
    else:
        qc = QuantumCircuit(*qc_slice.qregs, name=inst_name)
        for _ in range(num_time_slices * power):
            qc.append(qc_slice, qc.qubits)
            if barrier:
                qc.barrier(state_registers)
    return qc.to_instruction()
