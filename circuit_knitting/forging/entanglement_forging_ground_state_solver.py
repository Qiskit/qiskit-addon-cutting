# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the entanglement forging ground state solver class and associated support methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import (
    Callable,
    Iterable,
    Sequence,
    Union,
)

import scipy
import numpy as np

from qiskit.algorithms.optimizers import SPSA, Optimizer, OptimizerResult
from qiskit_nature.second_q.problems import (
    ElectronicStructureProblem,
    EigenstateResult,
    ElectronicBasis,
)
from qiskit_ibm_runtime import QiskitRuntimeService, Options
from qiskit.quantum_info import SparsePauliOp

from .entanglement_forging_ansatz import EntanglementForgingAnsatz
from .entanglement_forging_knitter import EntanglementForgingKnitter
from .entanglement_forging_operator import EntanglementForgingOperator
from .cholesky_decomposition import cholesky_decomposition, convert_cholesky_operator


OBJECTIVE = Callable[[np.ndarray], float]
MINIMIZER = Callable[
    [
        OBJECTIVE,  # the objective function to minimize
        np.ndarray,  # the initial point for the optimization
    ],
    Union[scipy.optimize.OptimizeResult, OptimizerResult],
]


@dataclass
class EntanglementForgingEvaluation:
    """Entanglement Forging Evaluation."""

    parameters: Sequence[float]
    eigenvalue: float
    eigenstate: np.ndarray


@dataclass
class EntanglementForgingHistory:
    """Entanglement Forging History."""

    evaluations: list[EntanglementForgingEvaluation] = field(default_factory=list)
    optimal_evaluation: EntanglementForgingEvaluation | None = None

    def store_evaluation(self, evaluation: EntanglementForgingEvaluation):
        """Store an evaluation iteration and update optimal values, as necessary."""
        self.evaluations.append(evaluation)
        if (
            self.optimal_evaluation is None
            or evaluation.eigenvalue < self.optimal_evaluation.eigenvalue
        ):
            self.optimal_evaluation = evaluation


class EntanglementForgingResult(EigenstateResult):
    """Entanglement Forging Result."""

    def __init__(self) -> None:
        """Initialize `EigenstateResult` parent class and set class fields."""
        super().__init__()
        self._energy_shift: float = 0.0
        self._history: list[EntanglementForgingEvaluation] = []

    @property
    def history(self) -> list[EntanglementForgingEvaluation]:
        """Return optimizer history."""
        return self._history

    @history.setter
    def history(self, history: list[EntanglementForgingEvaluation]) -> None:
        """Set optimizer history."""
        self._history = history

    @property
    def energy_shift(self) -> float:
        """Return the energy shift."""
        return self._energy_shift

    @energy_shift.setter
    def energy_shift(self, value: float) -> None:
        """Set the energy shift."""
        self._energy_shift = value

    @property
    def elapsed_time(self) -> float:
        """Return the elapsed time."""
        return self._elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, value: float):
        """Set the elapsed time."""
        self._elapsed_time = value


class EntanglementForgingGroundStateSolver:
    """A class which estimates the ground state energy of a molecule."""

    def __init__(
        self,
        ansatz: EntanglementForgingAnsatz | None = None,
        service: QiskitRuntimeService | None = None,
        optimizer: Optimizer | MINIMIZER | None = None,
        initial_point: np.ndarray | None = None,
        orbitals_to_reduce: Sequence[int] | None = None,
        backend_names: str | list[str] | None = None,
        options: Options | list[Options] | None = None,
        mo_coeff: np.ndarray | None = None,
        hf_energy: float | None = None,
    ):
        """
        Assign the necessary class variables and initialize any defaults.

        Args:
            ansatz: Class which holds the ansatz circuit and bitstrings
            service: The service used to spawn Qiskit primitives
            optimizer: Optimizer to use to optimize the ansatz circuit parameters
            initial_point: Initial values for ansatz parameters
            orbitals_to_reduce: List of orbital indices to remove from the problem before
                decomposition.
            backend_names: Backend name or list of backend names to use during parallel computation
            options: Options or list of options to be applied to the backends
            mo_coeff: Coefficients for converting an input problem to MO basis
            hf_energy: If set, this energy will be used instead of calculating the Hartree-Fock
                energy at each iteration.

        Returns:
            None
        """
        # Set class fields
        self._knitter: EntanglementForgingKnitter | None = None
        self._history: EntanglementForgingHistory = EntanglementForgingHistory()
        self._energy_shift = 0.0
        self._ansatz: EntanglementForgingAnsatz | None = ansatz
        self._service: QiskitRuntimeService | None = service
        self._initial_point: np.ndarray | None = initial_point
        self._orbitals_to_reduce = orbitals_to_reduce
        self.backend_names = backend_names  # type: ignore
        self.options = options
        self._mo_coeff = mo_coeff
        self._hf_energy = hf_energy
        self._optimizer: Optimizer | MINIMIZER = optimizer or SPSA()

    @property
    def ansatz(self) -> EntanglementForgingAnsatz | None:
        """Return ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: EntanglementForgingAnsatz | None) -> None:
        """Set the ansatz."""
        self._ansatz = ansatz

    @property
    def service(self) -> QiskitRuntimeService | None:
        """Return service."""
        return self._service

    @service.setter
    def service(self, service: QiskitRuntimeService | None) -> None:
        """Set the service."""
        self._service = service

    @property
    def orbitals_to_reduce(self) -> Sequence[int] | None:
        """Return the orbitals to reduce."""
        return self._orbitals_to_reduce

    @orbitals_to_reduce.setter
    def orbitals_to_reduce(self, orbitals_to_reduce: Sequence[int] | None) -> None:
        """Set the orbitals to reduce."""
        self._orbitals_to_reduce = orbitals_to_reduce

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer."""
        self._optimizer = optimizer

    @property
    def initial_point(self) -> np.ndarray | None:
        """Return the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray | None) -> None:
        """Set the initial point."""
        self._initial_point = initial_point

    @property
    def backend_names(self) -> list[str] | None:
        """Return the backend names."""
        return self._backend_names

    @backend_names.setter
    def backend_names(self, backend_names: str | list[str]) -> None:
        """Set the backend names."""
        if isinstance(backend_names, str):
            self._backend_names = [backend_names]
        else:
            self._backend_names = backend_names

    @property
    def options(self) -> list[Options] | None:
        """Return the options."""
        return self._options

    @options.setter
    def options(self, options: Options | list[Options]) -> None:
        """Set the options."""
        if isinstance(options, Options):
            self._options = [options]
        else:
            self._options = options

    @property
    def mo_coeff(self) -> np.ndarray | None:
        """Return the coefficients for converting integrals to the MO basis."""
        return self._mo_coeff

    @mo_coeff.setter
    def mo_coeff(self, mo_coeff: np.ndarray | None) -> None:
        """Set the coefficients for converting integrals to the MO basis."""
        self._mo_coeff = mo_coeff

    @property
    def hf_energy(self) -> float | None:
        """Return the Hartree-Fock energy."""
        return self._hf_energy

    @hf_energy.setter
    def hf_energy(self, hf_energy: float | None) -> None:
        """Set the Hartree-Fock energy."""
        self._hf_energy = hf_energy

    def solve(
        self,
        problem: ElectronicStructureProblem,
    ) -> EntanglementForgingResult:
        """Compute ground state properties.

        Args:
            problem: A class encoding a problem to be solved

        Returns:
            A result object

        Raises:
            ValueError: The ``backend_names`` and ``options`` lists are of
                incompatible lengths
            AttributeError: Ansatz must be set before calling `solve` method
        """
        if self._backend_names and self._options:
            if len(self._backend_names) != len(self._options):
                if len(self._options) == 1:
                    self._options = [self._options[0]] * len(self._backend_names)
                else:
                    raise AttributeError(
                        f"The list of backend names is length ({len(self._backend_names)}), "
                        f"but the list of options is length ({len(self._options)}). It is "
                        "ambiguous how to combine the options with the backends."
                    )
        if self._ansatz is None:
            raise AttributeError("Ansatz must be set before calling solve.")
        if self._initial_point is None:
            self._initial_point = np.array(
                [0.0 for i in range(len(self._ansatz.circuit_u.parameters))]
            )

        # Get the decomposed hamiltonian
        hamiltonian_terms = self.get_qubit_operators(problem)
        ef_operator = convert_cholesky_operator(hamiltonian_terms, self._ansatz)

        # Set the knitter class field
        if self._service is not None:
            backend_names = self._backend_names or ["ibmq_qasm_simulator"]
            self._knitter = EntanglementForgingKnitter(
                self._ansatz,
                hf_energy=self._hf_energy - self._energy_shift,
                service=self._service,
                backend_names=backend_names,
                options=self._options,
            )
        else:
            self._knitter = EntanglementForgingKnitter(
                self._ansatz, hf_energy=self._hf_energy, options=self._options
            )
        self._history = EntanglementForgingHistory()
        self._eval_count = 0
        evaluate_eigenvalue = self.get_eigenvalue_evaluation(ef_operator)

        # Minimize the minimum eigenvalue with respect to the ansatz parameters
        start_time = time()
        if callable(self._optimizer):
            self.optimizer(fun=evaluate_eigenvalue, x0=self._initial_point)
        else:
            self.optimizer.minimize(fun=evaluate_eigenvalue, x0=self._initial_point)
        elapsed_time = time() - start_time

        # Find the minimum eigenvalue found during optimization
        optimal_evaluation = self._history.optimal_evaluation
        if optimal_evaluation is None:
            raise RuntimeError("Unable to retrieve optimal evaluation.")

        # Create the EntanglementForgingResult from the results from the
        # results of eigenvalue minimization and other meta information
        result = EntanglementForgingResult()
        result.eigenvalues = np.asarray([optimal_evaluation.eigenvalue])
        result.eigenstates = [(optimal_evaluation.eigenstate, None)]
        result.history = self._history.evaluations
        result.energy_shift = self._energy_shift
        result.elapsed_time = elapsed_time

        # Close any runtime sessions
        self._knitter.close_sessions()

        return result

    def get_eigenvalue_evaluation(
        self, operator: EntanglementForgingOperator
    ) -> Callable[[Sequence[float]], float | Iterable[float]]:
        """
        Produce a callable which provides an estimation of the min eigenvalue of an operator.

        Args:
            operator: The decomposed Hamiltonian in entanglement forging format
        Returns:
            Callable function which provides an estimation of the mihnimum eigenvalue
            of the input operator given some ansatz circuit parameters.
        """

        def evaluate_eigenvalue(parameters: Sequence[float]) -> float:
            if self._knitter is None:
                raise RuntimeError("Knitter must be set before evaluating eigenvalue.")

            eigenvalue, schmidt_coeffs, _ = self._knitter(
                ansatz_parameters=parameters, forged_operator=operator
            )
            self._history.store_evaluation(
                EntanglementForgingEvaluation(parameters, eigenvalue, schmidt_coeffs)
            )

            return eigenvalue

        return evaluate_eigenvalue

    def get_qubit_operators(
        self,
        problem: ElectronicStructureProblem,
    ) -> list[SparsePauliOp]:
        """Construct decomposed qubit operators from an ``ElectronicStructureProblem``.

        Args:
            problem: A class encoding a problem to be solved

        Returns:
            hamiltonian_ops: Qubit operator representing the decomposed Hamiltonian

        Raises:
            ValueError: The input problem is not in MO basis, and ``mo_coeff`` is set to ``None``
            ValueError: The ``mo_coeff`` and input problem integrals are of incompatible shapes
            ValueError: The input integrals are ``None``
        """
        self._validate_problem_and_coeffs(problem, self.mo_coeff)

        hamiltonian_ops, self._energy_shift = cholesky_decomposition(
            problem, self.mo_coeff, self._orbitals_to_reduce  # type: ignore
        )
        return hamiltonian_ops

    @staticmethod
    def _validate_problem_and_coeffs(
        problem: ElectronicStructureProblem, mo_coeff: np.ndarray | None
    ):
        """Ensure the input problem can be translated to the MO basis."""
        if (problem.basis != ElectronicBasis.MO) and (mo_coeff is None):
            raise ValueError(
                "Cannot transform integrals to MO basis. The input problem is "
                f"in the ({problem.basis}) basis, and the mo_coeff class field is None."
            )

        h1 = np.array(problem.hamiltonian.electronic_integrals.one_body.alpha["+-"])
        if h1.shape == ():
            raise ValueError("The input integrals are empty.")

        # First two lines of this conditional are already implied by passing above checks, but alas, mypy :)
        if (
            problem.basis != ElectronicBasis.MO
            and mo_coeff is not None
            and mo_coeff.shape != h1.shape
        ):
            raise ValueError(
                f"The mo_coeff class field has shape ({mo_coeff.shape}), but "
                f"the input one body integral has shape ({h1.shape})."
            )
