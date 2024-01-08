"""File containing the wrapper class for optimizing LO gate and wire cuts."""
from itertools import count
from .cut_optimization import CutOptimization
from .cut_optimization import disjoint_subcircuit_actions
from .cut_optimization import CutOptimizationNextStateFunc
from .cut_optimization import CutOptimizationGoalStateFunc
from .cut_optimization import CutOptimizationMinCostBoundFunc
from .cut_optimization import CutOptimizationUpperBoundCostFunc
from .search_space_generator import SearchFunctions, SearchSpaceGenerator


### Functions for generating the cut optimization search space
cut_optimization_search_funcs = SearchFunctions(
    cost_func=CutOptimizationUpperBoundCostFunc,  # Change to CutOptimizationCostFunc with LOCC
    # or after the new LO QPD's are incorporated into CKT.
    upperbound_cost_func=CutOptimizationUpperBoundCostFunc,
    next_state_func=CutOptimizationNextStateFunc,
    goal_state_func=CutOptimizationGoalStateFunc,
    mincost_bound_func=CutOptimizationMinCostBoundFunc,
)


class LOCutsOptimizer:

    """Wrapper class for optimizing circuit cuts for the case in which
    only LO quasiprobability decompositions are employed.

    The search_engine_config dictionary that configures the optimization
    algorithms must be specified in the constructor.  For flexibility, the
    circuit_interface, optimization_settings, and device_constraints can
    be specified either in the constructor or in the optimize() method. In
    the latter case, the values provided overwrite the previous values.

    The circuit_interface object that is passed to the optimize()
    method is updated to reflect the optimized circuit cuts that were
    identified.

    Member Variables:

    circuit_interface (CircuitInterface) defines the circuit to be cut.

    optimization_settings (OptimizationSettings) defines the settings
    to be used for the optimization.

    device_constraints (DeviceConstraints) defines the capabilties of
    the target quantum hardware.

    search_engine_config (dict) maps names of stages of optimization to
    the corresponding SearchSpaceGenerator functions and actions that
    are used to perform the search for each stage.

    cut_optimization (CutOptimization) is the object created to
    perform the circuit cutting optimization.

    best_result (DisjointSubcircuitsState) is the lowest-cost
    DisjointSubcircuitsState object identified in the search.
    """

    def __init__(
        self,
        circuit_interface=None,
        optimization_settings=None,
        device_constraints=None,
        search_engine_config={
            "CutOptimization": SearchSpaceGenerator(
                functions=cut_optimization_search_funcs,
                actions=disjoint_subcircuit_actions,
            )
        },
    ):
        self.circuit_interface = circuit_interface
        self.optimization_settings = optimization_settings
        self.device_constraints = device_constraints
        self.search_engine_config = search_engine_config

        self.cut_optimization = None
        self.best_result = None

    def optimize(
        self,
        circuit_interface=None,
        optimization_settings=None,
        device_constraints=None,
    ):
        """Method to optimize the cutting of a circuit.

        Input Arguments:

        circuit_interface (CircuitInterface) defines the circuit to be
        cut.  This object is then updated with the optimized cuts that
        were identified.

        optimization_settings (OptimizationSettings) defines the settings
        to be used for the optimization.

        device_constraints (DeviceConstraints) defines the capabilties of
        the target quantum hardware.

        Returns:

        The lowest-cost DisjointSubcircuitsState object identified in
        the search, or None if no solution could be found.  In the
        case of the former, the circuit_interface object is also
        updated as a side effect to incorporate the cuts found.
        """

        if circuit_interface is not None:
            self.circuit_interface = circuit_interface

        if optimization_settings is not None:
            self.optimization_settings = optimization_settings

        if device_constraints is not None:
            self.device_constraints = device_constraints

        assert self.circuit_interface is not None, "circuit_interface cannot be None"

        assert (
            self.optimization_settings is not None
        ), "optimization_settings cannot be None"

        assert self.device_constraints is not None, "device_constraints cannot be None"

        # Perform cut optimization assuming no qubit reuse
        self.cut_optimization = CutOptimization(
            self.circuit_interface,
            self.optimization_settings,
            self.device_constraints,
            search_engine_config=self.search_engine_config,
        )

        out_1 = list()

        while True:
            state, cost = self.cut_optimization.optimizationPass()
            if state is None:
                break
            out_1.append((cost, state))

        min_cost = min(out_1, key=lambda x: x[0], default=None)

        if min_cost is not None:
            self.best_result = min_cost[-1]
            self.best_result.exportCuts(self.circuit_interface)
        else:
            self.best_result = None

        return self.best_result

    def getResults(self):
        """Return the optimization results."""

        return self.best_result

    def getStats(self, penultimate=False):
        """Return the optimization results."""

        return {
            "CutOptimization": self.cut_optimization.getStats(penultimate=penultimate)
        }

    def minimumReached(self):
        """Return a Boolean flag indicating whether the global
        minimum was reached.
        """

        return self.cut_optimization.minimumReached()


def printStateList(state_list):
    for x in state_list:
        print()
        x.print(simple=True)
