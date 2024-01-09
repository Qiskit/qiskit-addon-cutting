"""File containing the classes needed to generate and explore a search space."""


class ActionNames:

    """Class that maps action names to individual action objects
    and group names and to lists of action objects, where the
    action objects are used to generate a search space.

    Member Variables:

    action_dict (dict) maps action names to action objects.

    group_dict (dict) maps group names to lists of action objects.
    """

    def __init__(self):
        self.action_dict = dict()
        self.group_dict = dict()

    def copy(self, list_of_groups=None):
        """Return a copy of self that contains only those actions
        whose group affiliations intersect with list_of_groups.
        The default is to return a copy containing all actions.
        """

        action_list = getActionSubset(self.action_dict.values(), list_of_groups)

        new_container = ActionNames()
        for action in action_list:
            new_container.defineAction(action)

        return new_container

    def defineAction(self, action_object):
        """Insert the specified action object into the look-up
        dictionaries using the name of the action and its group
        names.
        """

        assert (
            action_object.getName() not in self.action_dict
        ), f"Action {action_object.getName()} is already defined"

        self.action_dict[action_object.getName()] = action_object

        group_name = action_object.getGroupNames()

        if isinstance(group_name, list) or isinstance(group_name, tuple):
            for name in group_name:
                if name not in self.group_dict:
                    self.group_dict[name] = list()
                self.group_dict[name].append(action_object)
        else:
            if group_name not in self.group_dict:
                self.group_dict[group_name] = list()
            self.group_dict[group_name].append(action_object)

    def defineActionList(self, action_list):
        """Inserts the specified action object into the look-up
        dictionaries using the name of the action and its group
        names"""

        for action in action_list:
            self.defineAction(action)

    def getAction(self, action_name):
        """Return the action object associated with the specified name.
        None is returned if there is no associated action object.
        """

        if action_name in self.action_dict:
            return self.action_dict[action_name]
        return None

    def getGroup(self, group_name):
        """Return the list of action objects associated with the group_name.
        None is returned if there are no associated action objects.
        """

        if group_name in self.group_dict:
            return self.group_dict[group_name]
        return None

    def getGroupActionNames(self, group_name):
        """Return a list of the names of action objects associated with
        the group_name. None is returned if there are no associated action
        objects.
        """

        if group_name in self.group_dict:
            return [a.getName() for a in self.group_dict[group_name]]
        return None

    def getActionNameList(self):
        """Return a list of action names that have been defined."""

        return list(self.action_dict.keys())

    def getGroupNameList(self):
        """Return a list of group names that have been defined."""

        return list(self.group_dict.keys())


def getActionSubset(action_list, action_groups):
    """Return the subset of actions in action_list whose group affiliations
    intersect with action_groups.
    """

    if action_groups is None:
        return action_list

    if len(action_groups) <= 0:
        action_groups = [None]

    groups = set(action_groups)

    return [
        a for a in action_list if len(groups.intersection(set(a.getGroupNames()))) > 0
    ]


class SearchFunctions:

    """Container class for holding functions needed to generate and explore
    a search space.  In addition to the required input arguments, the function
    signatures are assumed to also allow additional input arguments that are
    needed to perform the corresponding computations.  In particular, an
    ActionNames object should be incorporated into the additional input
    arguments in order to generate next-states.  For simplicity, all search
    algorithms will assume that all search-space functions employ the same set
    of additional arguments.

    Member Variables:

    cost_func (lambda state, *args) is a function that computes cost values
    from search states.  The cost returned can be numeric or tuples of
    numerics.  In the latter case, lexicographical comparisons are performed
    per Python semantics.

    stratum_func (lambda state, *args) is a function that computes stratum
    identifiers from search states, which are then used to stratify the search
    space when stratified beam search is employed.  The stratum_func can be
    None, in which case each level of the search has only one stratum, which
    is then labeled None.

    greedy_bound_func (lambda current_best_cost, *args) can be either
    None or a function that computes upper bounds to costs that are used during
    the greedy depth-first phases of search.  If None is provided, the upper
    bound is taken to be infinity.  In greedy search, the search proceeds in a
    greedy best-first, depth-first fashion until either a goal state is reached,
    a deadend is reached, or the cost bound provided by the greedy_bound_func is
    exceeded.  In the latter two cases, the search backjumps to the lowest cost
    state in the search frontier and the search proceeds from there.  The
    inputs passed to the greedy_bound_func are the current lowest cost in the
    search frontier and the input arguments that were passed to the
    optimizationPass() method of the search algorithm.  If the greedy_bound_func
    simply returns current_best_cost, then the search behavior is equivalent to
    pure best-first search.  Returning None is equivalent to returning an
    infinite greedy bound, which produces a purely greedy best-first,
    depth-first search.

    next_state_func (lambda state, *args) is a function that returns a list
    of next states generated from the input state.  An ActionNames object
    should be incorporated into the additional input arguments in order to
    generate next-states.

    goal_state_func (lambda state, *args) is a function that returns True if
    the input state is a solution state of the search.

    upperbound_cost_func (lambda goal_state, *args) can either be None or a
    function that returns an upper bound to the optimal cost given a goal_state
    as input.  The upper bound is used to prune next-states from the search in
    subsequent calls to the optimizationPass() method of the search algorithm.
    If upperbound_cost_func is None, the cost of the goal_state as determined
    by cost_func is used as an upper bound to the optimal cost.  If the
    upperbound_cost_func returns None, the effect is equivalent to returning
    an infinite upper bound (i.e., no cost pruning is performed on subsequent
    calls to the optimizationPass method.

    mincost_bound_func (lambda *args) can either be None or a function that
    returns a cost bound that is compared to the minimum cost across all
    vertices in a search frontier.  If the minimum cost exceeds the min-cost
    bound, the search is terminated even if a goal state has not yet been found.
    Returning None is equivalent to returning an infinite min-cost bound (i.e.,
    min-cost checking is effectively not performed).  A mincost_bound_func that
    is None is likewise equivalent to an infinite min-cost bound.
    """

    def __init__(
        self,
        cost_func=None,
        stratum_func=None,
        greedy_bound_func=None,
        next_state_func=None,
        goal_state_func=None,
        upperbound_cost_func=None,
        mincost_bound_func=None,
    ):
        self.cost_func = cost_func
        self.stratum_func = stratum_func
        self.greedy_bound_func = greedy_bound_func
        self.next_state_func = next_state_func
        self.goal_state_func = goal_state_func
        self.upperbound_cost_func = upperbound_cost_func
        self.mincost_bound_func = mincost_bound_func


class SearchSpaceGenerator:

    """Container class for holding both the functions and the
    associated actions needed to generate and explore a search space.

    Member Variables:

    functions (SearchFunctions) is a container class that holds
    the functions needed to generate and explore a search space.

    actions (ActionNames) is a container class that holds the search
    action objects needed to generate and explore a search space.
    The actions are expected to be passed as arguments to the search
    functions by a search engine.
    """

    def __init__(self, functions=None, actions=None):
        self.functions = functions
        self.actions = actions
