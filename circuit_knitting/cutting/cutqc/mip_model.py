# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the tools to find and manage the cuts."""

from __future__ import annotations

from typing import Sequence, Any

import numpy as np

try:
    from docplex.mp.model import Model
    from docplex.mp.utils import DOcplexException
except ModuleNotFoundError as ex:  # pragma: no cover
    raise ModuleNotFoundError(
        "DOcplex is not installed.  For automatic cut finding to work, both "
        "DOcplex and cplex must be available."
    ) from ex


class MIPModel(object):
    """
    Class to contain the model that manages the cut MIP.

    This class represents circuit cutting as a Mixed Integer Programming (MIP) problem
    that can then be solved (provably) optimally using a MIP solver. This is integrated
    with CPLEX, a fast commercial solver sold by IBM. There are free and open source MIP
    solvers, but they come with substantial slowdowns (often many orders of magnitude).
    By representing the original circuit as a Directed Acyclic Graph (DAG), this class
    can find the optimal wire cuts in the circuit.
    """

    def __init__(
        self,
        n_vertices: int,
        edges: Sequence[tuple[int]],
        vertex_ids: dict[str, int],
        id_vertices: dict[int, str],
        num_subcircuit: int,
        max_subcircuit_width: int,
        max_subcircuit_cuts: int,
        max_subcircuit_size: int,
        num_qubits: int,
        max_cuts: int,
    ):
        """
        Initialize member variables.

        Args:
            n_vertices: The number of vertices in the circuit DAG
            edges: The list of edges of the circuit DAG
            n_edges: The number of edges
            vertex_ids: Dictionary mapping vertices (i.e. two qubit gates) to the vertex
                id (i.e. a number)
            id_vertices: The inverse dictionary of vertex_ids, which has keys of vertex ids
                and values of the vertices
            num_subcircuit: The number of subcircuits
            max_subcircuit_width: Maximum number of qubits per subcircuit
            max_subcircuit_cuts: Maximum number of cuts in each subcircuit
            max_subcircuit_size: Maximum number of gates in a subcircuit
            num_qubits: The number of qubits in the circuit
            max_cuts: The maximum total number of cuts

        Returns:
            None
        """
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.vertex_ids = vertex_ids
        self.id_vertices = id_vertices
        self.num_subcircuit = num_subcircuit
        self.max_subcircuit_width = max_subcircuit_width
        self.max_subcircuit_cuts = max_subcircuit_cuts
        self.max_subcircuit_size = max_subcircuit_size
        self.num_qubits = num_qubits
        self.max_cuts = max_cuts

        self.subcircuit_counter: dict[int, dict[str, Any]] = {}

        """
        Count the number of input qubits directly connected to each node
        """
        self.vertex_weight = {}
        for node in self.vertex_ids:
            qargs = node.split(" ")
            num_in_qubits = 0
            for qarg in qargs:
                if int(qarg.split("]")[1]) == 0:
                    num_in_qubits += 1
            self.vertex_weight[node] = num_in_qubits

        self.model = Model("docplex_cutter")
        self.model.log_output = False
        self._add_variables()
        self._add_constraints()

    def _add_variables(self) -> None:
        """Add the necessary variables to the CPLEX model."""
        """
        Indicate if a vertex is in some subcircuit
        """
        self.vertex_var = []
        for i in range(self.num_subcircuit):
            subcircuit_y = []
            for j in range(self.n_vertices):
                varName = "bin_sc_" + str(i) + "_vx_" + str(j)
                loc_var = self.model.binary_var(name=varName)
                subcircuit_y.append(loc_var)
            self.vertex_var.append(subcircuit_y)

        """
        Indicate if an edge has one and only one vertex in some subcircuit
        """
        self.edge_var = []
        for i in range(self.num_subcircuit):
            subcircuit_x = []
            for j in range(self.n_edges):
                varName = "bin_sc_" + str(i) + "_edg_" + str(j)
                loc_var = self.model.binary_var(name=varName)
                subcircuit_x.append(loc_var)
            self.edge_var.append(subcircuit_x)

        """
        Total number of cuts
        add 0.1 for numerical stability
        """
        self.num_cuts = self.model.integer_var(
            lb=0, ub=self.max_cuts + 0.1, name="num_cuts"
        )

        for subcircuit in range(self.num_subcircuit):
            self.subcircuit_counter[subcircuit] = {}

            self.subcircuit_counter[subcircuit][
                "original_input"
            ] = self.model.integer_var(
                lb=0,
                ub=self.max_subcircuit_width,
                name="original_input_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["rho"] = self.model.integer_var(
                lb=0, ub=self.max_subcircuit_width, name="rho_%d" % subcircuit
            )
            self.subcircuit_counter[subcircuit]["O"] = self.model.integer_var(
                lb=0, ub=self.max_subcircuit_width, name="O_%d" % subcircuit
            )
            self.subcircuit_counter[subcircuit]["d"] = self.model.integer_var(
                lb=0.1, ub=self.max_subcircuit_width, name="d_%d" % subcircuit
            )
            if self.max_subcircuit_size is not None:
                self.subcircuit_counter[subcircuit]["size"] = self.model.integer_var(
                    lb=0.1, ub=self.max_subcircuit_size, name="size_%d" % subcircuit
                )
            if self.max_subcircuit_cuts is not None:
                self.subcircuit_counter[subcircuit][
                    "num_cuts"
                ] = self.model.integer_var(
                    lb=0.1, ub=self.max_subcircuit_cuts, name="num_cuts_%d" % subcircuit
                )

            self.subcircuit_counter[subcircuit]["rho_qubit_product"] = []
            self.subcircuit_counter[subcircuit]["O_qubit_product"] = []
            for i in range(self.n_edges):
                edge_var_downstream_vertex_var_product = self.model.binary_var(
                    name="bin_edge_var_downstream_vertex_var_product_%d_%d"
                    % (subcircuit, i)
                )
                self.subcircuit_counter[subcircuit]["rho_qubit_product"].append(
                    edge_var_downstream_vertex_var_product
                )
                edge_var_upstream_vertex_var_product = self.model.binary_var(
                    name="bin_edge_var_upstream_vertex_var_product_%d_%d"
                    % (subcircuit, i)
                )
                self.subcircuit_counter[subcircuit]["O_qubit_product"].append(
                    edge_var_upstream_vertex_var_product
                )

            if subcircuit > 0:
                lb = 0
                ub = self.num_qubits + 2 * self.max_cuts + 1
                self.subcircuit_counter[subcircuit][
                    "build_cost_exponent"
                ] = self.model.integer_var(
                    lb=lb, ub=ub, name="build_cost_exponent_%d" % subcircuit
                )

    def _add_constraints(self) -> None:
        """Add all contraints and objectives to MIP model."""
        """
        each vertex in exactly one subcircuit
        """
        for v in range(self.n_vertices):
            ctName = "cons_vertex_" + str(v)
            self.model.add_constraint(
                self.model.sum(
                    self.vertex_var[i][v] for i in range(self.num_subcircuit)
                )
                == 1,
                ctname=ctName,
            )

        """
        edge_var=1 indicates one and only one vertex of an edge is in subcircuit
        edge_var[subcircuit][edge] = vertex_var[subcircuit][u] XOR vertex_var[subcircuit][v]
        """
        for i in range(self.num_subcircuit):
            for e in range(self.n_edges):
                if len(self.edges[e]) != 2:
                    raise ValueError(
                        "Edges should be length 2 sequences: {self.edges[e]}"
                    )
                u = self.edges[e][0]
                v = self.edges[e][-1]
                u_vertex_var = self.vertex_var[i][u]
                v_vertex_var = self.vertex_var[i][v]
                ctName = "cons_edge_" + str(e)
                self.model.add_constraint(
                    self.edge_var[i][e] - u_vertex_var - v_vertex_var <= 0,
                    ctname=ctName + "_1",
                )
                self.model.add_constraint(
                    self.edge_var[i][e] - u_vertex_var + v_vertex_var >= 0,
                    ctname=ctName + "_2",
                )
                self.model.add_constraint(
                    self.edge_var[i][e] - v_vertex_var + u_vertex_var >= 0,
                    ctname=ctName + "_3",
                )
                self.model.add_constraint(
                    self.edge_var[i][e] + u_vertex_var + v_vertex_var <= 2,
                    ctname=ctName + "_4",
                )

        """
        Symmetry-breaking constraints
        Force small-numbered vertices into small-numbered subcircuits:
            v0: in subcircuit 0
            v1: in subcircuit_0 or subcircuit_1
            v2: in subcircuit_0 or subcircuit_1 or subcircuit_2
            ...
        """
        for vertex in range(self.num_subcircuit):
            ctName = "cons_symm_" + str(vertex)
            self.model.add_constraint(
                self.model.sum(
                    self.vertex_var[subcircuit][vertex]
                    for subcircuit in range(vertex + 1)
                )
                == 1,
                ctname=ctName,
            )

        """
        Compute number of cuts
        """
        self.model.add_constraint(
            self.num_cuts
            == self.model.sum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                ]
            )
            / 2
        )

        num_effective_qubits = []
        for subcircuit in range(self.num_subcircuit):
            """
            Compute number of different types of qubit in a subcircuit
            """
            self.model.add_constraint(
                self.subcircuit_counter[subcircuit]["original_input"]
                - self.model.sum(
                    self.vertex_weight[self.id_vertices[i]]
                    * self.vertex_var[subcircuit][i]
                    for i in range(self.n_vertices)
                )
                == 0,
                ctname="cons_subcircuit_input_%d" % subcircuit,
            )

            for i in range(self.n_edges):
                if len(self.edges[i]) != 2:
                    raise ValueError(
                        "Edges should be length 2 sequences: {self.edges[i]}"
                    )
                self.model.add_constraint(
                    self.subcircuit_counter[subcircuit]["rho_qubit_product"][i]
                    == self.edge_var[subcircuit][i]
                    & self.vertex_var[subcircuit][self.edges[i][-1]],
                    ctname="cons_edge_var_downstream_vertex_var_%d_%d"
                    % (subcircuit, i),
                )

            self.model.add_constraint(
                self.subcircuit_counter[subcircuit]["rho"]
                - self.model.sum(
                    self.subcircuit_counter[subcircuit]["rho_qubit_product"]
                )
                == 0,
                ctname="cons_subcircuit_rho_qubits_%d" % subcircuit,
            )

            for i in range(self.n_edges):
                self.model.add_constraint(
                    self.subcircuit_counter[subcircuit]["O_qubit_product"][i]
                    == self.edge_var[subcircuit][i]
                    & self.vertex_var[subcircuit][self.edges[i][0]],
                    ctname="cons_edge_var_upstream_vertex_var_%d_%d" % (subcircuit, i),
                )
            self.model.add_constraint(
                self.subcircuit_counter[subcircuit]["O"]
                - self.model.sum(self.subcircuit_counter[subcircuit]["O_qubit_product"])
                == 0,
                ctname="cons_subcircuit_O_qubits_%d" % subcircuit,
            )

            self.model.add_constraint(
                self.subcircuit_counter[subcircuit]["d"]
                - self.subcircuit_counter[subcircuit]["original_input"]
                - self.subcircuit_counter[subcircuit]["rho"]
                == 0,
                ctname="cons_subcircuit_d_qubits_%d" % subcircuit,
            )

            if self.max_subcircuit_cuts is not None:
                self.model.add_constraint(
                    self.subcircuit_counter[subcircuit]["num_cuts"]
                    - self.subcircuit_counter[subcircuit]["rho"]
                    - self.subcircuit_counter[subcircuit]["O"]
                    == 0,
                    ctname="cons_subcircuit_num_cuts_%d" % subcircuit,
                )

            if self.max_subcircuit_size is not None:
                self.model.add_constraint(
                    self.subcircuit_counter[subcircuit]["size"]
                    - self.model.sum(
                        [self.vertex_var[subcircuit][v] for v in range(self.n_vertices)]
                    )
                    == 0,
                    ctname="cons_subcircuit_size_%d" % subcircuit,
                )

            num_effective_qubits.append(
                self.subcircuit_counter[subcircuit]["d"]
                - self.subcircuit_counter[subcircuit]["O"]
            )

            """
            Compute the classical postprocessing cost
            """
            if subcircuit > 0:
                ptx, ptf = self.pwl_exp(
                    lb=int(
                        self.subcircuit_counter[subcircuit]["build_cost_exponent"].lb
                    ),
                    ub=int(
                        self.subcircuit_counter[subcircuit]["build_cost_exponent"].ub
                    ),
                    base=2,
                    coefficient=1,
                    integer_only=True,
                )
                self.model.add_constraint(
                    self.subcircuit_counter[subcircuit]["build_cost_exponent"]
                    - self.model.sum(num_effective_qubits)
                    - 2 * self.num_cuts
                    == 0,
                    ctname="cons_build_cost_exponent_%d" % subcircuit,
                )
                # TODO: add PWL objective in CPLEX
                # self.model.setPWLObj(self.subcircuit_counter[subcircuit]['build_cost_exponent'], ptx, ptf)

        # self.model.setObjective(self.num_cuts,gp.GRB.MINIMIZE)
        self.model.set_objective("min", self.num_cuts)

    def pwl_exp(
        self, lb: int, ub: int, base: int, coefficient: int, integer_only: bool
    ) -> tuple[list[int], list[int]]:
        r"""
        Approximate a nonlinear exponential function via a piecewise linear function.

        Args:
            lb: Lower bound
            ub: Upper bound
            base: The base of the input exponential
            coefficient: The coefficient of the original exponential
            integer_only: Whether the input x's are only integers

        Returns:
            A tuple containing the :math:`x`\ s and :math:`f(x)`\ s of the piecewise approximation
        """
        # Piecewise linear approximation of coefficient*base**x
        ptx = []
        ptf = []

        x_range = range(lb, ub + 1) if integer_only else list(np.linspace(lb, ub, 200))
        # print('x_range : {}, integer_only : {}'.format(x_range,integer_only))
        for x in x_range:
            y = coefficient * base**x
            ptx.append(x)
            ptf.append(y)
        return ptx, ptf

    def check_graph(self, n_vertices: int, edges: Sequence[tuple[int]]) -> None:
        """
        Ensure circuit DAG is viable.

        This means that there are no oversized edges, that all edges are from viable nodes,
        and that the graph is otherwise a valid graph.

        Args:
            n_vertices: The number of vertices
            edges: The edge list

        Returns:
            None

        Raises:
            ValueError: The graph is invalid
        """
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])  # type: ignore
        vertices |= set([i for (_, i) in edges])  # type: ignore
        assert vertices == set(range(n_vertices))
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("Edges should be length 2 sequences: {edge}")
            u = edge[0]
            v = edge[-1]
            if u > v:
                raise ValueError(f"Edge u ({u}) cannot be greater than edge v ({v})")
            if u > n_vertices:
                raise ValueError(
                    f"Edge u ({u}) cannot be greater than number of vertices ({n_vertices})"
                )

    def solve(self, min_postprocessing_cost: float) -> bool:
        """
        Solve the MIP model.

        Args:
            min_post_processing_cost: The predicted minimum post-processing cost,
                often is inf

        Returns:
            Flag denoting whether or not the model found a solution
        """
        # print('solving for %d subcircuits'%self.num_subcircuit)
        # print('model has %d variables, %d linear constraints,%d quadratic constraints, %d general constraints'
        # % (self.model.NumVars,self.model.NumConstrs, self.model.NumQConstrs, self.model.NumGenConstrs))
        print(
            "Exporting as a LP file to let you check the model that will be solved : ",
            min_postprocessing_cost,
            str(type(min_postprocessing_cost)),
        )
        try:
            self.model.export_as_lp(path="./docplex_cutter.lp")
        except RuntimeError:
            print(
                "The LP file export has failed.  This is known to happen sometimes "
                "when cplex is not installed.  Now attempting to continue anyway."
            )
        try:
            self.model.set_time_limit(300)
            if min_postprocessing_cost != float("inf"):
                self.model.parameters.mip.tolerances.uppercutoff(
                    min_postprocessing_cost
                )
            self.model.solve(log_output=True)

        except DOcplexException as e:
            print("Caught: " + e.message)
            raise e

        if self.model._has_solution:
            my_solve_details = self.model.solve_details
            self.subcircuits = []
            self.optimal = my_solve_details.status == "optimal"
            self.runtime = my_solve_details.time
            self.node_count = my_solve_details.nb_nodes_processed
            self.mip_gap = my_solve_details.mip_relative_gap
            self.objective = self.model.objective_value

            for i in range(self.num_subcircuit):
                subcircuit = []
                for j in range(self.n_vertices):
                    if abs(self.vertex_var[i][j].solution_value) > 1e-4:
                        subcircuit.append(self.id_vertices[j])
                self.subcircuits.append(subcircuit)
            assert (
                sum([len(subcircuit) for subcircuit in self.subcircuits])
                == self.n_vertices
            )

            cut_edges_idx = []
            cut_edges = []
            for i in range(self.num_subcircuit):
                for j in range(self.n_edges):
                    if (
                        abs(self.edge_var[i][j].solution_value) > 1e-4
                        and j not in cut_edges_idx
                    ):
                        cut_edges_idx.append(j)
                        if len(self.edges[j]) != 2:
                            raise ValueError("Edges should be length-2 sequences.")
                        u = self.edges[j][0]
                        v = self.edges[j][-1]
                        cut_edges.append((self.id_vertices[u], self.id_vertices[v]))
            self.cut_edges = cut_edges
            return True
        else:
            return False
