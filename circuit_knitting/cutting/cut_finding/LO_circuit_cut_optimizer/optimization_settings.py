"""File containing class for specifying parameters that control the optimization."""


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

    gate_cut_LOCC_no_ancillas (bool) is a flag that indicates that
    LOCC gate cuts with no ancillas should be included in the optimization.

    wire_cut_LO (bool) is a flag that indicates that LO wire cuts should be
    included in the optimization.

    wire_cut_LOCC_with_ancillas (bool) is a flag that indicates that
    LOCC wire cuts with ancillas should be included in the optimization.

    wire_cut_LOCC_no_ancillas (bool) is a flag that indicates that
    LOCC wire cuts with no ancillas should be included in the optimization.

    Raises:

    ValueError: max_gamma must be a positive definite integer.
    ValueError: max_backjumps must be a positive semi-definite integer.
    ValueError: beam_width must be a positive definite integer.
    """

    def __init__(
        self,
        max_gamma=1024,
        max_backjumps=10000,
        greedy_multiplier=None,
        beam_width=30,
        rand_seed=None,
        LO=True,
        LOCC_ancillas=False,
        LOCC_no_ancillas=False,
        engine_selections={"PhaseOneStageOneNoQubitReuse": "Greedy"},
    ):
        if not (isinstance(max_gamma, int) and max_gamma > 0):
            raise ValueError("max_gamma must be a positive definite integer.")

        if not (isinstance(max_backjumps, int) and max_backjumps >= 0):
            raise ValueError("max_backjumps must be a positive semi-definite integer.")

        if not (isinstance(beam_width, int) and beam_width > 0):
            raise ValueError("beam_width must be a positive definite integer.")

        self.max_gamma = max_gamma
        self.max_backjumps = max_backjumps
        self.greedy_multiplier = greedy_multiplier
        self.beam_width = beam_width
        self.rand_seed = rand_seed
        self.engine_selections = engine_selections.copy()
        self.LO = LO
        self.LOCC_ancillas = LOCC_ancillas
        self.LOCC_no_ancillas = LOCC_no_ancillas

        self.gate_cut_LO = self.LO
        self.gate_cut_LOCC_with_ancillas = self.LOCC_ancillas
        self.gate_cut_LOCC_no_ancillas = self.LOCC_no_ancillas

        self.wire_cut_LO = self.LO
        self.wire_cut_LOCC_with_ancillas = self.LOCC_ancillas
        self.wire_cut_LOCC_no_ancillas = self.LOCC_no_ancillas

    def getMaxGamma(self):
        """Return the max gamma."""
        return self.max_gamma

    def getMaxBackJumps(self):
        """Return the maximum number of allowed search backjumps."""
        return self.max_backjumps

    def getGreedyMultiplier(self):
        """Return the greedy multiplier."""
        return self.greedy_multiplier

    def getBeamWidth(self):
        """Return the beam width."""
        return self.beam_width

    def getRandSeed(self):
        """Return the random seed."""
        return self.rand_seed

    def getEngineSelection(self, stage_of_optimization):
        """Return the name of the search engine to employ."""
        return self.engine_selections[stage_of_optimization]

    def setEngineSelection(self, stage_of_optimization, engine_name):
        """Return the name of the search engine to employ."""
        self.engine_selections[stage_of_optimization] = engine_name

    def clearAllCutTypes(self):
        """Reset the flags for all circuit cutting types"""

        self.gate_cut_LO = False
        self.gate_cut_LOCC_with_ancillas = False
        self.gate_cut_LOCC_no_ancillas = False

        self.wire_cut_LO = False
        self.wire_cut_LOCC_with_ancillas = False
        self.wire_cut_LOCC_no_ancillas = False

    def setGateCutTypes(self):
        """Select which gate-cut types to include in the optimization.
        The default is to include all gate-cut types.
        """

        self.gate_cut_LO = self.LO
        self.gate_cut_LOCC_with_ancillas = self.LOCC_ancillas
        self.gate_cut_LOCC_no_ancillas = self.LOCC_no_ancillas

    def setWireCutTypes(self):
        """Select which wire-cut types to include in the optimization.
        The default is to include all wire-cut types.
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
            or self.gate_cut_LOCC_no_ancillas
        ):
            out.append("GateCut")

        if (
            self.wire_cut_LO
            or self.wire_cut_LOCC_with_ancillas
            or self.wire_cut_LOCC_no_ancillas
        ):
            out.append("WireCut")

        return out
