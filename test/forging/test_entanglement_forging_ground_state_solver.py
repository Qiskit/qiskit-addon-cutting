# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for EntanglementForgingVQE module."""

import os
import unittest
import importlib.util

import numpy as np
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import COBYLA
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import (
    ElectronicBasis,
    ElectronicStructureProblem,
)
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.formats import get_ao_to_mo_from_qcschema
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

from circuit_knitting.forging import (
    EntanglementForgingAnsatz,
    EntanglementForgingGroundStateSolver,
)

pyscf_available = importlib.util.find_spec("pyscf") is not None


class TestEntanglementForgingGroundStateSolver(unittest.TestCase):
    def setUp(self):
        # Hard-code some ansatz params/lambdas
        self.optimizer = SPSA(maxiter=0)

    @unittest.skipIf(not pyscf_available, "pyscf is not installed")
    def test_entanglement_forging_vqe_hydrogen(self):
        """Test of applying Entanglement Forged Solver to to compute the energy of a H2 molecule."""
        # Set up the ElectronicStructureProblem
        driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735",
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )
        driver.run()
        problem = driver.to_problem(basis=ElectronicBasis.AO)
        qcschema = driver.to_qcschema()
        mo_coeff = get_ao_to_mo_from_qcschema(qcschema).coefficients.alpha["+-"]

        # Specify the ansatz and bitstrings
        ansatz = EntanglementForgingAnsatz(
            circuit_u=TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1),
            bitstrings_u=[(1, 0), (0, 1)],
        )

        # Set up the entanglement forging vqe object
        solver = EntanglementForgingGroundStateSolver(
            ansatz=ansatz,
            optimizer=self.optimizer,
            initial_point=[0.0, np.pi / 2],
            mo_coeff=mo_coeff,
        )

        # Solve for the ground state energy
        results = solver.solve(problem)
        ground_state_energy = results.groundenergy + results.energy_shift

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(ground_state_energy, -1.121936544469326)

    @unittest.skipIf(not pyscf_available, "pyscf is not installed")
    def test_fixed_hf_h2o(self):
        """Test for fixing the HF value in computing the energy of a H2O molecule."""
        # Set up the ElectronicStructureProblem
        HF = -14.09259461609392
        orb_act = list(range(0, 5))
        num_alpha = num_beta = 3
        hcore = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "H2O_one_body.npy"),
        )
        eri = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "H2O_two_body.npy"),
        )
        hamiltonian = ElectronicEnergy.from_raw_integrals(hcore, eri)
        hamiltonian.nuclear_repulsion_energy = -61.57756706745154
        problem = ElectronicStructureProblem(hamiltonian)
        problem.basis = ElectronicBasis.MO
        problem.num_particles = (num_alpha, num_beta)
        transformer = ActiveSpaceTransformer(
            num_electrons=6, num_spatial_orbitals=len(orb_act), active_orbitals=orb_act
        )
        problem_reduced = transformer.transform(problem)

        theta = Parameter("θ")

        hop_gate = QuantumCircuit(2, name="hop_gate")
        hop_gate.h(0)
        hop_gate.cx(1, 0)
        hop_gate.cx(0, 1)
        hop_gate.ry(-theta, 0)
        hop_gate.ry(-theta, 1)
        hop_gate.cx(0, 1)
        hop_gate.h(0)

        theta_1, theta_2, theta_3, theta_4 = (
            Parameter("θ1"),
            Parameter("θ2"),
            Parameter("θ3"),
            Parameter("θ4"),
        )

        circuit = QuantumCircuit(5)
        circuit.append(hop_gate.to_gate({theta: theta_1}), [0, 1])
        circuit.append(hop_gate.to_gate({theta: theta_2}), [3, 4])
        circuit.append(hop_gate.to_gate({theta: 0}), [1, 4])
        circuit.append(hop_gate.to_gate({theta: theta_3}), [0, 2])
        circuit.append(hop_gate.to_gate({theta: theta_4}), [3, 4])

        bitstrings = [(1, 1, 1, 0, 0), (0, 1, 1, 0, 1), (1, 1, 0, 1, 0)]
        ansatz = EntanglementForgingAnsatz(
            circuit_u=circuit,
            bitstrings_u=bitstrings,
        )

        optimizer = COBYLA(maxiter=1)
        initial_point = [-0.83604922, -0.87326138, -0.93964018, 0.55224467]
        solver = EntanglementForgingGroundStateSolver(
            ansatz=ansatz,
            optimizer=optimizer,
            hf_energy=HF,
            initial_point=initial_point,
        )
        result = solver.solve(problem_reduced)

        assert np.allclose(
            result.groundstate[0], [-0.00252657, 0.99945784, -0.03282741], atol=1e-8
        )
