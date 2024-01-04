"""File containing class for specifying parameters that control the optimization."""
class OptimizationSettings:

    """Class for specifying parameters that control the optimization.
    In this release, only LO gate cuts are supported. Other cut types,
    including "LOCC" gate and wire cuts will be added in future releases.

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

    rand_seed (int) is a seed used to provide a repeatable initialization
    of the pesudorandom number generators used by the optimization, which
    is useful for debugging purposes.  If None is used as the random seed,
    then a seed is obtained using an operating-system call to achieve an
    unrepeatable randomized initialization, which is useful in practice.

    gate_cut_LO (bool) is a flag that indicates that LO gate cuts should be
    included in the optimization.

    Raises:

    ValueError: max_gamma must be a positive definite integer.
    ValueError: max_backjumps must be a positive semi-definite integer.
    ValueError: beam_width must be a positive definite integer.
    """

    def __init__(
        self,
        max_gamma=1024,
        max_backjumps=10000,
        beam_width=30,
        rand_seed=None,
        LO=True,
        engine_selections={"CutOptimization": "BestFirst"},
    ):
        if not (isinstance(max_gamma, int) and max_gamma > 0):
            raise ValueError("max_gamma must be a positive definite integer.")

        if not (isinstance(max_backjumps, int) and max_backjumps >= 0):
            raise ValueError("max_backjumps must be a positive semi-definite integer.")

        if not (isinstance(beam_width, int) and beam_width > 0):
            raise ValueError("beam_width must be a positive definite integer.")

        self.max_gamma = max_gamma
        self.max_backjumps = max_backjumps
        self.beam_width = beam_width
        self.rand_seed = rand_seed
        self.engine_selections = engine_selections.copy()
        self.gate_cut_LO = LO

    def getMaxGamma(self):
        """Return the max gamma."""
        return self.max_gamma

    def getMaxBackJumps(self):
        """Return the maximum number of allowed search backjumps."""
        return self.max_backjumps

    def getBeamWidth(self):
        """Return the beam width."""
        return self.beam_width

    def getRandSeed(self):
        """Return the random seed."""
        return self.rand_seed

    def setEngineSelection(self, stage_of_optimization, engine_name):
        """Return the name of the search engine to employ."""
        self.engine_selections[stage_of_optimization] = engine_name

    def getEngineSelection(self, stage_of_optimization):
        """Return the name of the search engine to employ."""
        return self.engine_selections[stage_of_optimization]

    def clearAllCutTypes(self):
        """Reset the flags for all cut types. In this release, only LO gate
        cuts are supported. Other cut types, including "LOCC" gate and wire cuts will
        be added in future releases.
        """

        self.gate_cut_LO = False

    def setGateCutTypes(self):
        """Select which gate-cut types to include in the optimization.
        The default is to include all gate-cut types. In this release, only LO gate
        cuts are supported. Other cut types, including "LOCC" gate and wire cuts will
        be added in future releases.
        """

        self.gate_cut_LO

    def getCutSearchGroups(self):
        """Return a list of search-action groups to include in the
        optimization for cutting circuits into disjoint subcircuits. In this release,
        only LO gate cuts are supported. Other cut types, including "LOCC" gate and
        wire cuts will be added in future releases.
        """

        out = [None]

        if self.gate_cut_LO:
            out.append("GateCut")

        return out
