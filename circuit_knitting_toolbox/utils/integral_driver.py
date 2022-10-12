# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""IntegralDriver."""
import numpy as np
from nptyping import Float, Int, NDArray, Shape
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriver
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    ElectronicStructureDriverResult,
    ParticleNumber,
)
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

SingleBodyIntegrals = NDArray[Shape["N, N"], Float]
TwoBodyIntegrals = NDArray[Shape["N, N, N, N"], Float]


class IntegralDriver(ElectronicStructureDriver):
    """IntegralDriver."""

    def __init__(
        self,
        hcore: SingleBodyIntegrals,
        mo_coeff: NDArray[Shape["N, N"], Float],
        eri: TwoBodyIntegrals,
        num_alpha: int,
        num_beta: int,
        nuclear_repulsion_energy: float,
    ):
        """Entanglement forging driver.

        Args:
            hcore: hcore integral
            mo_coeff: MO coefficients
            eri: eri integral
            num_alpha: number of alpha electrons
            num_beta: number of beta electrons
            nuclear_repulsion_energy: nuclear repulsion energy
        """
        super().__init__()

        self._hcore = hcore
        self._mo_coeff = mo_coeff
        self._eri = eri
        self._num_alpha = num_alpha
        self._num_beta = num_beta
        self._nuclear_repulsion_energy = nuclear_repulsion_energy

    def run(self) -> ElectronicStructureDriverResult:
        """Return ElectronicStructureDriverResult constructed from input data."""
        # Create ParticleNumber property. Multiply by 2 since the number
        # of spin orbitals is 2x the number of MOs
        particle_number = ParticleNumber(
            self._mo_coeff.shape[0] * 2, (self._num_alpha, self._num_beta)
        )

        # Define the transform from AO to MO
        elx_basis_xform = ElectronicBasisTransform(
            ElectronicBasis.AO, ElectronicBasis.MO, self._mo_coeff
        )

        # One and two-body integrals in AO basis
        one_body_ao = OneBodyElectronicIntegrals(
            ElectronicBasis.AO, (self._hcore, None)
        )
        two_body_ao = TwoBodyElectronicIntegrals(
            ElectronicBasis.AO, (self._eri, None, None, None)
        )

        # One and two-body integrals in MO basis
        one_body_mo = one_body_ao.transform_basis(elx_basis_xform)
        two_body_mo = two_body_ao.transform_basis(elx_basis_xform)

        # Instantiate ElectronicEnergy property object
        electronic_energy = ElectronicEnergy(
            [one_body_ao, two_body_ao, one_body_mo, two_body_mo],
            nuclear_repulsion_energy=self._nuclear_repulsion_energy,
        )

        result = ElectronicStructureDriverResult()
        result.add_property(electronic_energy)
        result.add_property(particle_number)
        result.add_property(elx_basis_xform)

        return result
