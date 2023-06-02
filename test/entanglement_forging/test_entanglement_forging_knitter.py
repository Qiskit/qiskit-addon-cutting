# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for EntanglementForgingKnitter module."""

import os
import unittest
import pytest

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem, ElectronicBasis
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.formats import get_ao_to_mo_from_qcschema

from circuit_knitting_toolbox.entanglement_forging import (
    EntanglementForgingAnsatz,
    EntanglementForgingKnitter,
    cholesky_decomposition,
    convert_cholesky_operator,
)


class TestEntanglementForgingKnitter(unittest.TestCase):
    def setUp(self):
        self.energy_shift_h2 = 0.7199689944489797

        self.energy_shift_h2o = -61.37433060587931

        self.hcore_o2 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "O2_one_body.npy"),
        )
        self.eri_o2 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "O2_two_body.npy"),
        )
        self.energy_shift_o2 = -99.83894101027317

        self.hcore_ch3 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CH3_one_body.npy"),
        )
        self.eri_ch3 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CH3_two_body.npy"),
        )
        self.energy_shift_ch3 = -31.90914780401554

        self.hcore_cn = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CN_one_body.npy"),
        )
        self.eri_cn = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CN_two_body.npy"),
        )
        self.energy_shift_cn = -67.18561556466743

    def create_mock_ansatz_circuit(self, num_qubits: int) -> QuantumCircuit:
        theta = Parameter("θ")
        mock_gate = QuantumCircuit(1, name="mock gate")
        mock_gate.rz(theta, 0)

        theta_vec = [Parameter("θ%d" % i) for i in range(1)]
        ansatz = QuantumCircuit(num_qubits)
        ansatz.append(mock_gate.to_gate({theta: theta_vec[0]}), [0])

        return ansatz

    def test_entanglement_forging_H2(self):
        """
        Test to apply Entanglement Forging to compute the energy of a H2 molecule,
        given the optimial ansatz parameters.
        """
        # Set up the ELectrionicStructureProblem
        driver = PySCFDriver("H 0.0 0.0 0.0; H 0.0 0.0 0.735")
        driver.run()
        problem = driver.to_problem(basis=ElectronicBasis.AO)
        qcschema = driver.to_qcschema()

        # Specify the ansatz and bitstrings
        ansatz = EntanglementForgingAnsatz(
            circuit_u=TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1),
            bitstrings_u=[(1, 0), (0, 1), (1, 0)],
            bitstrings_v=[(1, 0), (0, 1), (0, 1)],
        )

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)

        # Specify the decomposition method and get the forged operator
        mo_coeff = get_ao_to_mo_from_qcschema(qcschema).coefficients.alpha["+-"]
        hamiltonian_terms, energy_shift = cholesky_decomposition(
            problem, mo_coeff=mo_coeff
        )
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Hard-coded optimal ansatz parameters
        ansatz_params = [0, 1.57079633]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -1.121936544469326)

    def test_entanglement_forging_H2O(self):  # pylint: disable=too-many-locals
        """
        Test to apply Entanglement Forging to compute the energy of a H2O molecule,
        given optimal ansatz parameters.
        """
        # setup problem
        radius_1 = 0.958  # position for the first H atom
        radius_2 = 0.958  # position for the second H atom
        thetas_in_deg = 104.478  # bond angles.

        h1_x = radius_1
        h2_x = radius_2 * np.cos(np.pi / 180 * thetas_in_deg)
        h2_y = radius_2 * np.sin(np.pi / 180 * thetas_in_deg)

        molecule = f"O 0.0 0.0 0.0; H {h1_x} 0.0 0.0; H {h2_x} {h2_y} 0.0"
        driver = PySCFDriver(molecule, basis="sto6g")
        driver.run()
        problem = driver.to_problem(basis=ElectronicBasis.AO)
        qcschema = driver.to_qcschema()

        # solution
        orbitals_to_reduce = [0, 3]
        bitstrings = [
            (1, 1, 1, 1, 1, 0, 0),
            (1, 0, 1, 1, 1, 0, 1),
            (1, 0, 1, 1, 1, 1, 0),
        ]
        reduced_bitstrings = [
            tuple(bs)
            for bs in np.delete(bitstrings, orbitals_to_reduce, axis=-1).tolist()
        ]

        theta = Parameter("θ")
        theta_1, theta_2, theta_3, theta_4 = (
            Parameter("θ1"),
            Parameter("θ2"),
            Parameter("θ3"),
            Parameter("θ4"),
        )

        hop_gate = QuantumCircuit(2, name="Hop gate")
        hop_gate.h(0)
        hop_gate.cx(1, 0)
        hop_gate.cx(0, 1)
        hop_gate.ry(-theta, 0)
        hop_gate.ry(-theta, 1)
        hop_gate.cx(0, 1)
        hop_gate.h(0)

        circuit = QuantumCircuit(5)
        circuit.append(hop_gate.to_gate({theta: theta_1}), [0, 1])
        circuit.append(hop_gate.to_gate({theta: theta_2}), [3, 4])
        circuit.append(hop_gate.to_gate({theta: 0}), [1, 4])
        circuit.append(hop_gate.to_gate({theta: theta_3}), [0, 2])
        circuit.append(hop_gate.to_gate({theta: theta_4}), [3, 4])

        ansatz = EntanglementForgingAnsatz(
            circuit_u=circuit, bitstrings_u=reduced_bitstrings
        )

        # Specify the decomposition method and get the forged operator
        mo_coeff = get_ao_to_mo_from_qcschema(qcschema).coefficients.alpha["+-"]
        hamiltonian_terms, energy_shift = cholesky_decomposition(
            problem, mo_coeff=mo_coeff, orbitals_to_reduce=orbitals_to_reduce
        )
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)

        ansatz_params = [0.0, 0.0, 0.0, 0.0]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -75.68366174497027)

    def test_entanglement_forging_driver_H2(self):
        """Test for entanglement forging driver."""
        hcore = np.array([[-1.12421758, -0.9652574], [-0.9652574, -1.12421758]])
        mo_coeff = np.array([[0.54830202, 1.21832731], [0.54830202, -1.21832731]])
        eri = np.array(
            [
                [
                    [[0.77460594, 0.44744572], [0.44744572, 0.57187698]],
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                ],
                [
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                    [[0.57187698, 0.44744572], [0.44744572, 0.77460594]],
                ],
            ]
        )

        hamiltonian = ElectronicEnergy.from_raw_integrals(hcore, eri)
        hamiltonian.nuclear_repulsion_energy = 0.7199689944489797
        problem = ElectronicStructureProblem(hamiltonian)
        problem.num_particles = (1, 1)
        problem.basis = ElectronicBasis.AO

        ansatz = EntanglementForgingAnsatz(
            bitstrings_u=[(1, 0), (0, 1)],
            circuit_u=TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1),
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(
            problem, mo_coeff=mo_coeff
        )
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)
        ansatz_params = [0.0, np.pi / 2]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -1.1219365445030705)

    @pytest.mark.slow
    def test_asymmetric_bitstrings_O2(self):
        """Test for entanglement forging driver."""
        hamiltonian = ElectronicEnergy.from_raw_integrals(self.hcore_o2, self.eri_o2)
        hamiltonian.nuclear_repulsion_energy = self.energy_shift_o2
        problem = ElectronicStructureProblem(hamiltonian)
        problem.num_particles = (6, 6)

        ansatz = EntanglementForgingAnsatz(
            circuit_u=self.create_mock_ansatz_circuit(8),
            bitstrings_u=[
                (1, 1, 1, 1, 1, 1, 0, 0),
                (1, 1, 1, 1, 1, 0, 1, 0),
                (1, 1, 1, 1, 0, 1, 1, 0),
                (1, 1, 0, 1, 1, 1, 1, 0),
            ],
            bitstrings_v=[
                (1, 1, 1, 1, 1, 0, 1, 0),
                (1, 1, 1, 1, 1, 1, 0, 0),
                (1, 1, 0, 1, 1, 1, 1, 0),
                (1, 1, 1, 1, 0, 1, 1, 0),
            ],
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(problem)
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)
        ansatz_params = [0.0]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -147.63645235088566)

    def test_asymmetric_bitstrings_CH3(self):
        """Test for entanglement forging driver."""
        hamiltonian = ElectronicEnergy.from_raw_integrals(self.hcore_ch3, self.eri_ch3)
        hamiltonian.nuclear_repulsion_energy = self.energy_shift_ch3
        problem = ElectronicStructureProblem(hamiltonian)
        problem.num_particles = (3, 2)

        ansatz = EntanglementForgingAnsatz(
            circuit_u=self.create_mock_ansatz_circuit(6),
            bitstrings_u=[
                (1, 1, 1, 0, 0, 0),
                (0, 1, 1, 0, 0, 1),
                (1, 0, 1, 0, 1, 0),
                (1, 0, 1, 1, 0, 0),
                (0, 1, 1, 1, 0, 0),
            ],
            bitstrings_v=[
                (1, 1, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 1),
                (1, 0, 0, 0, 1, 0),
                (1, 0, 0, 1, 0, 0),
                (0, 1, 0, 1, 0, 0),
            ],
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(problem)
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)
        ansatz_params = [0.0]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -39.09031477502881)

    @pytest.mark.slow
    def test_asymmetric_bitstrings_CN(self):
        """Test for asymmetric bitstrings with hybrid cross terms."""
        hamiltonian = ElectronicEnergy.from_raw_integrals(self.hcore_cn, self.eri_cn)
        hamiltonian.nuclear_repulsion_energy = self.energy_shift_cn
        problem = ElectronicStructureProblem(hamiltonian)
        problem.num_particles = (5, 4)

        ansatz = EntanglementForgingAnsatz(
            circuit_u=self.create_mock_ansatz_circuit(8),
            bitstrings_u=[
                (1, 1, 1, 1, 1, 0, 0, 0),
                (1, 1, 1, 0, 1, 0, 1, 0),
                (1, 1, 0, 1, 1, 1, 0, 0),
                (1, 1, 1, 0, 1, 1, 0, 0),
                (1, 1, 0, 1, 1, 0, 1, 0),
                (1, 1, 1, 1, 1, 0, 0, 0),
            ],
            bitstrings_v=[
                (1, 1, 1, 1, 0, 0, 0, 0),
                (1, 1, 1, 0, 0, 0, 1, 0),
                (1, 1, 0, 1, 0, 1, 0, 0),
                (1, 1, 1, 0, 0, 1, 0, 0),
                (1, 1, 0, 1, 0, 0, 1, 0),
                (1, 0, 1, 1, 1, 0, 0, 0),
            ],
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(problem)
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)
        ansatz_params = [0.0]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -91.06680541685226)
