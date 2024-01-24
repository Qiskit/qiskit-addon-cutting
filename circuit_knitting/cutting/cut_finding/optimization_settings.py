# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for specifying parameters that control the optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizationSettings:
    """Class for specifying parameters that control the optimization.

    Member Variables:

    max_gamma (int) is a constraint on the maximum value of gamma that a
    solution to the optimization is allowed to have to be considered feasible.
    All other potential solutions are discarded.

    engine_selections (dict) is a dictionary that defines the selections
    of search engines for the various stages of optimization. In this release
    only "BestFirst" or Dijkstra's best-first search is supported. In future
    relesases the choices "Greedy" and "BeamSearch", which correspond respectively
    to bounded-greedy and best-first search and beam search will be added.

    max_backjumps (int) is a constraint on the maximum number of backjump
    operations that can be performed by the search algorithm.  This constraint
    does not apply to beam search.

    beam_width (int) is the beam width used in the optimization.  Only the B
    best partial solutions are maintained at each level in the search, where B
    is the beam width.  This constraint only applies to beam search algorithms.

    greedy_multiplier (float) is a multiplier used to compute cost bounds
    for bounded-greedy best-first search.

    rand_seed (int) is a seed used to provide a repeatable initialization
    of the pesudorandom number generators used by the optimization, which
    is useful for debugging purposes.  If None is used as the random seed,
    then a seed is obtained using an operating-system call to achieve an
    unrepeatable randomized initialization, which is useful in practice.

    gate_cut_LO (bool) is a flag that indicates that LO gate cuts should be
    included in the optimization.

    gate_cut_LOCC_with_ancillas (bool) is a flag that indicates that
    LOCC gate cuts with ancillas should be included in the optimization.

    wire_cut_LO (bool) is a flag that indicates that LO wire cuts should be
    included in the optimization.

    wire_cut_LOCC_with_ancillas (bool) is a flag that indicates that
    LOCC wire cuts with ancillas should be included in the optimization.

    wire_cut_LOCC_no_ancillas (bool) is a flag that indicates that
    LOCC wire cuts with no ancillas should be included in the optimization.

    NOTE: The current release only supports LO gate and wire cuts. LOCC
    flags have been incorporated with an eye towards future releases.

    Raises:

    ValueError: max_gamma must be a positive definite integer.
    ValueError: max_backjumps must be a positive semi-definite integer.
    ValueError: beam_width must be a positive definite integer.
    """

    max_gamma: int = 1024
    max_backjumps: int = 10000
    rand_seed: int | None = None
    LO: bool = True
    LOCC_ancillas: bool = False
    LOCC_no_ancillas: bool = False
    engine_selections: dict[str, str] | None = None

    def __post_init__(self):
        if self.max_gamma < 1:
            raise ValueError("max_gamma must be a positive definite integer.")
        if self.max_backjumps < 0:
            raise ValueError("max_backjumps must be a positive semi-definite integer.")

        self.gate_cut_LO = self.LO
        self.gate_cut_LOCC_with_ancillas = self.LOCC_ancillas
        self.gate_cut_LOCC_no_ancillas = self.LOCC_no_ancillas

        self.wire_cut_LO = self.LO
        self.wire_cut_LOCC_with_ancillas = self.LOCC_ancillas
        self.wire_cut_LOCC_no_ancillas = self.LOCC_no_ancillas
        if self.engine_selections is None:
            self.engine_selections = {"CutOptimization": "BestFirst"}

    def getMaxGamma(self):
        """Return the max gamma."""
        return self.max_gamma

    def getMaxBackJumps(self):
        """Return the maximum number of allowed search backjumps."""
        return self.max_backjumps

    def getRandSeed(self):
        """Return the random seed."""
        return self.rand_seed

    def getEngineSelection(self, stage_of_optimization):
        """Return the name of the search engine to employ."""
        return self.engine_selections[stage_of_optimization]

    def setEngineSelection(self, stage_of_optimization, engine_name):
        """Return the name of the search engine to employ."""
        self.engine_selections[stage_of_optimization] = engine_name

    def setGateCutTypes(self):
        """Select which gate-cut types to include in the optimization.
        The default is to include LO gate cuts.
        """
        self.gate_cut_LO = self.LO
        self.gate_cut_LOCC_with_ancillas = self.LOCC_ancillas

    def setWireCutTypes(self):
        """Select which wire-cut types to include in the optimization.
        The default is to include LO wire cuts.
        """

        self.wire_cut_LO = self.LO
        self.wire_cut_LOCC_with_ancillas = self.LOCC_ancillas
        self.wire_cut_LOCC_no_ancillas = self.LOCC_no_ancillas

    def getCutSearchGroups(self):
        """Return a list of search-action groups to include in the
        optimization for cutting circuits into disjoint subcircuits.
        """

        out = [None]

        if (
            self.gate_cut_LO
            or self.gate_cut_LOCC_with_ancillas
        ):
            out.append("GateCut")

        if (
            self.wire_cut_LO
            or self.wire_cut_LOCC_with_ancillas
            or self.wire_cut_LOCC_no_ancillas
        ):
            out.append("WireCut")

        return out

    @classmethod
    def from_dict(cls, options: dict[str, int]):
        return cls(**options)
