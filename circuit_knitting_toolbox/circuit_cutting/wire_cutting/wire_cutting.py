"""File containing the tools to find and manage the cuts."""
import math
import typing
from typing import Sequence, Dict, Tuple, Union, Any, Optional, List, cast


from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException
from docplex.mp.sdetails import SolveDetails
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit


class MIP_Model(object):
    """
    Class to contain the model that manages the cut MIP.
    
    This class represents circuit cutting as a Mixed Integer Programming (MIP) problem
    that can then be solved (provably) optimally using a MIP solver. This is integrated
    with CPLEX, a fast commercial solver sold by IBM. There are free and open source MIP
    solvers, but they come with substantial slowdowns (often many orders of magnitude).
    By representing the original circuit as a Directed Acyclic Graph (DAG), this class
    can find the optimal wire cuts in the circuit.

    Attributes:
        - n_vertices (int): the number of vertices in the circuit DAG
        - edges (list): the list of edges of the circuit DAG
        - n_edges (int): the number of edges
        - vertex_ids (dict): dictionary mapping vertices (i.e. two qubit gates) to the vertex
            id (i.e. a number)
        - id_vertices (dict): the inverse dictionary of vertex_ids, which has keys of vertex ids
            and values of the vertices
        - num_subcircuit (int): the number of subcircuits
        - max_subcircuit_width (int): maximum number of qubits per subcircuit
        - max_subcircuit_cuts (int): maximum number of cuts in each subcircuit
        - max_subcircuit_size (int): maximum number of gates in a subcircuit
        - num_qubits (int): the number of qubits in the circuit
        - max_cuts (int): the maximum total number of cuts
        - subcircuit_counter (dict): a tracker for the information regarding subcircuits
        - vertex_weight (dict): keep track of the number of input qubits directly connected
            to each node
        - model (docplex model): the model interface for CPLEX
    """

    def __init__(
        self,
        n_vertices: int,
        edges: Sequence[Tuple[int]],
        vertex_ids: Dict[str, int],
        id_vertices: Dict[int, str],
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
            - n_vertices (int): the number of vertices in the circuit DAG
            - edges (list): the list of edges of the circuit DAG
            - n_edges (int): the number of edges
            - vertex_ids (dict): dictionary mapping vertices (i.e. two qubit gates) to the vertex
                id (i.e. a number)
            - id_vertices (dict): the inverse dictionary of vertex_ids, which has keys of vertex ids
                and values of the vertices
            - num_subcircuit (int): the number of subcircuits
            - max_subcircuit_width (int): maximum number of qubits per subcircuit
            - max_subcircuit_cuts (int): maximum number of cuts in each subcircuit
            - max_subcircuit_size (int): maximum number of gates in a subcircuit
            - num_qubits (int): the number of qubits in the circuit
            - max_cuts (int): the maximum total number of cuts

        Returns:
            - None
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

        self.subcircuit_counter: Dict[int, Dict[str, Any]] = {}

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
        """
        Add the necessary variables to the CPLEX model.

        Args:
            - self

        Returns:
            - None
        """
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
        """
        Add all contraints and objectives to MIP model.
        
        Args:
            - self
        Returns:
            - None
        """
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
    ) -> Tuple[List[int], List[int]]:
        """
        Approximate a nonlinear exponential function via a piecewise linear function.

        Args:
            - lb (int): lower bound
            - ub (int): upper bound
            - base (int): the base of the input exponential
            - coefficient (int): the coefficient of the original exponential
            - integer_only (bool): whether the input x's are only integers
        
        Returns:
            - (list): the x's of the piecewise approximation
            - (list): the f(x)'s of the piecewise approximation
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

    def check_graph(self, n_vertices: int, edges: Sequence[Tuple[int]]) -> None:
        """
        Ensure circuit DAG is viable.
        
        This means that there are no oversized edges, that all edges are from viable nodes,
        and that the graph is otherwise a valid graph.

        Args:
            - n_vertices (int): the number of vertices
            - edges (list): the edge list

        Returns:
            - None

        Raises:
            - ValueError: if the graph is invalid
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
            - min_post_processing_cost (float): the predicted minimum post-processing cost, 
                often is inf

        Returns:
            - (bool): whether or not the model found a solution
        """
        # print('solving for %d subcircuits'%self.num_subcircuit)
        # print('model has %d variables, %d linear constraints,%d quadratic constraints, %d general constraints'
        # % (self.model.NumVars,self.model.NumConstrs, self.model.NumQConstrs, self.model.NumGenConstrs))
        print(
            "Exporting as a LP file to let you check the model that will be solved : ",
            min_postprocessing_cost,
            str(type(min_postprocessing_cost)),
        )
        self.model.export_as_lp(path="./docplex_cutter.lp")
        try:
            self.model.set_time_limit(300)
            if min_postprocessing_cost != float("inf"):
                self.model.parameters.mip.tolerances.uppercutoff(
                    min_postprocessing_cost
                )
            self.model.solve(log_output=True)

        except DOcplexException as e:
            print("Caught: " + e.message)

        if self.model._has_solution:
            my_solve_details = self.model.solve_details
            self.objective = None
            self.subcircuits = []
            self.optimal = self.model.get_solve_status() == "optimal"
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


def read_circuit(
    circuit: QuantumCircuit,
) -> Tuple[int, List[Tuple[int, int]], Dict[str, int], Dict[int, str]]:
    """
    Read the input circuit to a graph based representation for the MIP model.

    Args:
        - circuit (QuantumCircuit): a stripped circuit to be converted into a
            DAG like representation
    
    Returns:
        - (int): number of vertices
        - (list): edge list
        - (dict): the dictionary mapping vertices to vertex numbers
        - (dict): the dictionary mapping vertex numbers to vertex information
    """
    dag = circuit_to_dag(circuit)
    edges = []
    node_name_ids = {}
    id_node_names = {}
    vertex_ids = {}
    curr_node_id = 0
    qubit_gate_counter = {}
    for qubit in dag.qubits:
        qubit_gate_counter[qubit] = 0
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) != 2:
            raise Exception("vertex does not have 2 qargs!")
        arg0, arg1 = vertex.qargs
        vertex_name = "%s[%d]%d %s[%d]%d" % (
            arg0.register.name,
            arg0.index,
            qubit_gate_counter[arg0],
            arg1.register.name,
            arg1.index,
            qubit_gate_counter[arg1],
        )
        qubit_gate_counter[arg0] += 1
        qubit_gate_counter[arg1] += 1
        # print(vertex.op.label,vertex_name,curr_node_id)
        if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
            node_name_ids[vertex_name] = curr_node_id
            id_node_names[curr_node_id] = vertex_name
            vertex_ids[id(vertex)] = curr_node_id
            curr_node_id += 1

    for u, v, _ in dag.edges():
        if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):
            u_id = vertex_ids[id(u)]
            v_id = vertex_ids[id(v)]
            edges.append((u_id, v_id))

    n_vertices = dag.size()

    return n_vertices, edges, node_name_ids, id_node_names


def cuts_parser(
    cuts: Sequence[Tuple[str]], circ: QuantumCircuit
) -> List[Tuple[Qubit, int]]:
    """
    Convert cuts to wires.

    Args:
        - cuts (list): the cuts found by the model (or provided by the user)
        - circ (QuantumCircuit): the quantum circuit the cuts are from

    Returns:
        - (list): the list containing the wires that were cut and the gates
            that are affected by these cuts
    """
    dag = circuit_to_dag(circ)
    positions = []
    for position in cuts:
        if len(position) != 2:
            raise ValueError(
                "position variable should be a length 2 sequence: {position}"
            )
        source = position[0]
        dest = position[-1]
        source_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in source.split(" ")
        ]
        dest_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in dest.split(" ")
        ]
        qubit_cut = []
        for source_qarg in source_qargs:
            source_qubit, source_multi_Q_gate_idx = source_qarg
            for dest_qarg in dest_qargs:
                dest_qubit, dest_multi_Q_gate_idx = dest_qarg
                if (
                    source_qubit == dest_qubit
                    and dest_multi_Q_gate_idx == source_multi_Q_gate_idx + 1
                ):
                    qubit_cut.append(source_qubit)
        # if len(qubit_cut)>1:
        #     raise Exception('one cut is cutting on multiple qubits')
        for x in source.split(" "):
            if x.split("]")[0] + "]" == qubit_cut[0]:
                source_idx = int(x.split("]")[1])
        for x in dest.split(" "):
            if x.split("]")[0] + "]" == qubit_cut[0]:
                dest_idx = int(x.split("]")[1])
        multi_Q_gate_idx = max(source_idx, dest_idx)

        wire = None
        for qubit in circ.qubits:
            if qubit.register.name == qubit_cut[0].split("[")[0] and qubit.index == int(
                qubit_cut[0].split("[")[1].split("]")[0]
            ):
                wire = qubit
        tmp = 0
        all_Q_gate_idx = None
        for gate_idx, gate in enumerate(
            list(dag.nodes_on_wire(wire=wire, only_ops=True))
        ):
            if len(gate.qargs) > 1:
                tmp += 1
                if tmp == multi_Q_gate_idx:
                    all_Q_gate_idx = gate_idx
        if (wire is None) or (all_Q_gate_idx is None):
            raise ValueError("Something unexpected happened while parsing cuts.")
        positions.append((wire, all_Q_gate_idx))
    positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions


def subcircuits_parser(
    subcircuit_gates: List[List[str]], circuit: QuantumCircuit
) -> Tuple[Sequence[QuantumCircuit], Dict[Qubit, List[Dict[str, Union[int, Qubit]]]]]:
    """
    Convert the subcircuit gates into quantum circuits and path out the DAGs to enable conversion.
    
    Args:
        - subcircuit_gates (list): the gates in the subcircuits
        - circuit (QuantumCircuit): the original circuit

    Returns:
        - (list): the subcircuits
        - (dict): the paths in the quantum circuit DAGs
    """
    """
    Assign the single qubit gates to the closest two-qubit gates
    """

    def calculate_distance_between_gate(gate_A, gate_B):
        if len(gate_A.split(" ")) >= len(gate_B.split(" ")):
            tmp_gate = gate_A
            gate_A = gate_B
            gate_B = tmp_gate
        distance = float("inf")
        for qarg_A in gate_A.split(" "):
            qubit_A = qarg_A.split("]")[0] + "]"
            qgate_A = int(qarg_A.split("]")[-1])
            for qarg_B in gate_B.split(" "):
                qubit_B = qarg_B.split("]")[0] + "]"
                qgate_B = int(qarg_B.split("]")[-1])
                # print('%s gate %d --> %s gate %d'%(qubit_A,qgate_A,qubit_B,qgate_B))
                if qubit_A == qubit_B:
                    distance = min(distance, abs(qgate_B - qgate_A))
        # print('Distance from %s to %s = %f'%(gate_A,gate_B,distance))
        return distance

    dag = circuit_to_dag(circuit)
    qubit_allGate_depths = {x: 0 for x in circuit.qubits}
    qubit_2qGate_depths = {x: 0 for x in circuit.qubits}
    gate_depth_encodings = {}
    # print('Before translation :',subcircuit_gates,flush=True)
    for op_node in dag.topological_op_nodes():
        gate_depth_encoding = ""
        for qarg in op_node.qargs:
            gate_depth_encoding += "%s[%d]%d " % (
                qarg.register.name,
                qarg.index,
                qubit_allGate_depths[qarg],
            )
        gate_depth_encoding = gate_depth_encoding[:-1]
        gate_depth_encodings[op_node] = gate_depth_encoding
        for qarg in op_node.qargs:
            qubit_allGate_depths[qarg] += 1
        if len(op_node.qargs) == 2:
            MIP_gate_depth_encoding = ""
            for qarg in op_node.qargs:
                MIP_gate_depth_encoding += "%s[%d]%d " % (
                    qarg.register.name,
                    qarg.index,
                    qubit_2qGate_depths[qarg],
                )
                qubit_2qGate_depths[qarg] += 1
            MIP_gate_depth_encoding = MIP_gate_depth_encoding[:-1]
            # print('gate_depth_encoding = %s, MIP_gate_depth_encoding = %s'%(gate_depth_encoding,MIP_gate_depth_encoding))
            for subcircuit_idx in range(len(subcircuit_gates)):
                for gate_idx in range(len(subcircuit_gates[subcircuit_idx])):
                    if (
                        subcircuit_gates[subcircuit_idx][gate_idx]
                        == MIP_gate_depth_encoding
                    ):
                        subcircuit_gates[subcircuit_idx][gate_idx] = gate_depth_encoding
                        break
    # print('After translation :',subcircuit_gates,flush=True)
    subcircuit_op_nodes: Dict[int, List[DAGOpNode]] = {
        x: [] for x in range(len(subcircuit_gates))
    }
    subcircuit_sizes = [0 for x in range(len(subcircuit_gates))]
    complete_path_map: Dict[Qubit, List[Dict[str, Union[int, Qubit]]]] = {}
    for circuit_qubit in dag.qubits:
        complete_path_map[circuit_qubit] = []
        qubit_ops = dag.nodes_on_wire(wire=circuit_qubit, only_ops=True)
        for qubit_op_idx, qubit_op in enumerate(qubit_ops):
            gate_depth_encoding = gate_depth_encodings[qubit_op]
            nearest_subcircuit_idx = -1
            min_distance = float("inf")
            for subcircuit_idx in range(len(subcircuit_gates)):
                distance = float("inf")
                for gate in subcircuit_gates[subcircuit_idx]:
                    if len(gate.split(" ")) == 1:
                        # Do not compare against single qubit gates
                        continue
                    else:
                        distance = min(
                            distance,
                            calculate_distance_between_gate(
                                gate_A=gate_depth_encoding, gate_B=gate
                            ),
                        )
                # print('Distance from %s to subcircuit %d = %f'%(gate_depth_encoding,subcircuit_idx,distance))
                if distance < min_distance:
                    min_distance = distance
                    nearest_subcircuit_idx = subcircuit_idx
            assert nearest_subcircuit_idx != -1
            path_element = {
                "subcircuit_idx": nearest_subcircuit_idx,
                "subcircuit_qubit": subcircuit_sizes[nearest_subcircuit_idx],
            }
            if (
                len(complete_path_map[circuit_qubit]) == 0
                or nearest_subcircuit_idx
                != complete_path_map[circuit_qubit][-1]["subcircuit_idx"]
            ):
                # print('{} op #{:d} {:s} encoding = {:s}'.format(circuit_qubit,qubit_op_idx,qubit_op.name,gate_depth_encoding),
                # 'belongs in subcircuit %d'%nearest_subcircuit_idx)
                complete_path_map[circuit_qubit].append(path_element)
                subcircuit_sizes[nearest_subcircuit_idx] += 1

            subcircuit_op_nodes[nearest_subcircuit_idx].append(qubit_op)
    for circuit_qubit in complete_path_map:
        # print(circuit_qubit,'-->')
        for path_element in complete_path_map[circuit_qubit]:
            path_element_qubit = QuantumRegister(
                size=subcircuit_sizes[path_element["subcircuit_idx"]], name="q"
            )[path_element["subcircuit_qubit"]]
            path_element["subcircuit_qubit"] = path_element_qubit
            # print(path_element)
    subcircuits = generate_subcircuits(
        subcircuit_op_nodes=subcircuit_op_nodes,
        complete_path_map=complete_path_map,
        subcircuit_sizes=subcircuit_sizes,
        dag=dag,
    )
    return subcircuits, complete_path_map


def generate_subcircuits(
    subcircuit_op_nodes: Dict[int, List[DAGOpNode]],
    complete_path_map: Dict[Qubit, List[Dict[str, Union[int, Qubit]]]],
    subcircuit_sizes: Sequence[int],
    dag: DAGCircuit,
) -> Sequence[QuantumCircuit]:
    """
    Generate the subcircuits from given nodes and paths.

    Called in the subcircuit_parser function to convert the found paths and nodes
    into actual quantum circuit objects.
    
    Args:
        - subcircuit_op_nodes (dict): the nodes of each of the subcircuits
        - complete_path_map (dict): the complete path through the subcircuits
        - subcircuit_sizes (list): the number of qubits in each of the subcircuits
        - dag (DAGCircuit): the dag representation of the input quantum circuit

    Returns:
        - (list): the subcircuits
    """
    qubit_pointers = {x: 0 for x in complete_path_map}
    subcircuits = [QuantumCircuit(x, name="q") for x in subcircuit_sizes]
    for op_node in dag.topological_op_nodes():
        subcircuit_idx_list = list(
            filter(
                lambda x: op_node in subcircuit_op_nodes[x], subcircuit_op_nodes.keys()
            )
        )
        if len(subcircuit_idx_list) != 1:
            raise ValueError("A node cannot belong to more than one subcircuit.")
        subcircuit_idx = subcircuit_idx_list[0]
        # print('{} belongs in subcircuit {:d}'.format(op_node.qargs,subcircuit_idx))
        subcircuit_qargs = []
        for op_node_qarg in op_node.qargs:
            if (
                complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]][
                    "subcircuit_idx"
                ]
                != subcircuit_idx
            ):
                qubit_pointers[op_node_qarg] += 1
            path_element = complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]]
            assert path_element["subcircuit_idx"] == subcircuit_idx
            subcircuit_qargs.append(path_element["subcircuit_qubit"])
        # print('-->',subcircuit_qargs)

        # mypy doesn't recognize QuantumCircuit as being an Iterable, so we ignore
        subcircuits[subcircuit_idx].append(  # type: ignore
            instruction=op_node.op, qargs=subcircuit_qargs, cargs=None
        )
    return subcircuits


def circuit_stripping(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Remove all single qubit and barrier type gates.

    Args:
        - circuit (QuantumCircuit): the circuit to strip
    
    Returns:
        - (QuantumCircuit): the stripped circuit
    """
    # Remove all single qubit gates and barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) == 2 and vertex.op.name != "barrier":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)


def cost_estimate(counter: Dict[int, Dict[str, int]]) -> float:
    """
    Estimate the cost of processing the subcircuits.
    
    Args:
        - counter (dict): dictionary containing information for each of the
            subcircuits

    Returns:
        - (float): the estimated cost for classical processing
    """
    num_cuts = sum([counter[subcircuit_idx]["rho"] for subcircuit_idx in counter])
    subcircuit_indices = list(counter.keys())
    num_effective_qubits_list = [
        counter[subcircuit_idx]["effective"] for subcircuit_idx in subcircuit_indices
    ]
    num_effective_qubits, _ = zip(
        *sorted(zip(num_effective_qubits_list, subcircuit_indices))
    )
    classical_cost = 0
    accumulated_kron_len = 2 ** num_effective_qubits[0]
    for effective in num_effective_qubits[1:]:
        accumulated_kron_len *= 2**effective
        classical_cost += accumulated_kron_len
    classical_cost *= 4**num_cuts
    return classical_cost


def get_pairs(
    complete_path_map: Dict[Qubit, List[Dict[str, Union[int, Qubit]]]]
) -> List[Tuple[Dict[str, Union[int, Qubit]], Dict[str, Union[int, Qubit]]]]:
    """
    Get all pairs through each path.
    
    Iterates through the path for each of the qubits and keeps track of the
    each pair of neigbors. 
    
    Args:
        - complete_path_map (dict): the dictionary containing all path information
    
    Returns:
        - (list): all pairs for each of the qubit paths
    """
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path) > 1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr + 1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs


def get_counter(
    subcircuits: Sequence[QuantumCircuit],
    O_rho_pairs: List[
        Tuple[Dict[str, Union[int, Qubit]], Dict[str, Union[int, Qubit]]]
    ],
) -> Dict[int, Dict[str, int]]:
    """
    Create information regarding each of the subcircuit parameters (qubits, width, etc.).

    Args:
        - subcircuits (list): the list of subcircuits
        - O_rho_pairs (list): the pairs for each qubit path as generated in the get_pairs
            function
    
    Returns:
        - (dict): the resulting dictionary with all parameter information
    """
    counter = {}
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        counter[subcircuit_idx] = {
            "effective": subcircuit.num_qubits,
            "rho": 0,
            "O": 0,
            "d": subcircuit.num_qubits,
            "depth": subcircuit.depth(),
            "size": subcircuit.size(),
        }
    for pair in O_rho_pairs:
        Sequence[Tuple[Dict[str, Union[int, Qubit]]]]
        if len(pair) != 2:
            raise ValueError(f"O_rho_pairs must be length 2: {pair}")
        O_qubit = pair[0]
        rho_qubit = pair[-1]
        counter[O_qubit["subcircuit_idx"]]["effective"] -= 1
        counter[O_qubit["subcircuit_idx"]]["O"] += 1
        counter[rho_qubit["subcircuit_idx"]]["rho"] += 1
    return counter


@typing.no_type_check
def find_wire_cuts(
    circuit: QuantumCircuit,
    max_subcircuit_width: int,
    max_cuts: Optional[int],
    num_subcircuits: Optional[Sequence[int]],
    max_subcircuit_cuts: Optional[int],
    max_subcircuit_size: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    """
    Find optimal cuts for the wires.

    Will print if the model cannot find a solution at all, and will print whether
    the found solution is optimal or not.

    Args:
        - circuit (QuantumCircuit): original quantum circuit to be cut into subcircuits
        - max_subcircuit_width (int): max number of qubits in each subcircuit
        - max_cuts (int, optional): max total number of cuts allowed
        - num_subcircuits (list, optional): list of number of subcircuits to try
        - max_subcircuit_cuts (int, optional): max number of cuts for a subcircuit
        - max_subcircuit_size (int, optional): the maximum number of two qubit gates in each
            subcircuit
        - verbose (bool): whether to print information about the cut finding or not

    Returns:
        - (dict): the solution found for the cuts
    """
    stripped_circ = circuit_stripping(circuit=circuit)
    n_vertices, edges, vertex_ids, id_vertices = read_circuit(circuit=stripped_circ)
    num_qubits = circuit.num_qubits
    cut_solution = {}
    min_cost = float("inf")

    best_mip_model = None
    for num_subcircuit in num_subcircuits:
        if (
            num_subcircuit * max_subcircuit_width - (num_subcircuit - 1) < num_qubits
            or num_subcircuit > num_qubits
            or max_cuts + 1 < num_subcircuit
        ):
            if verbose:
                print("%d subcircuits : IMPOSSIBLE" % (num_subcircuit))
            continue
        kwargs = dict(
            n_vertices=n_vertices,
            edges=edges,
            vertex_ids=vertex_ids,
            id_vertices=id_vertices,
            num_subcircuit=num_subcircuit,
            max_subcircuit_width=max_subcircuit_width,
            max_subcircuit_cuts=max_subcircuit_cuts,
            max_subcircuit_size=max_subcircuit_size,
            num_qubits=num_qubits,
            max_cuts=max_cuts,
        )

        mip_model = MIP_Model(**kwargs)
        feasible = mip_model.solve(min_postprocessing_cost=min_cost)
        if not feasible:
            if verbose:
                print("%d subcircuits : NO SOLUTIONS" % (num_subcircuit))
            continue
        else:
            min_objective = mip_model.objective
            positions = cuts_parser(mip_model.cut_edges, circuit)
            subcircuits, complete_path_map = subcircuits_parser(
                subcircuit_gates=mip_model.subcircuits, circuit=circuit
            )
            O_rho_pairs = get_pairs(complete_path_map=complete_path_map)
            counter = get_counter(subcircuits=subcircuits, O_rho_pairs=O_rho_pairs)

            classical_cost = cost_estimate(counter=counter)
            cost = classical_cost

            if cost < min_cost:
                min_cost = cost
                best_mip_model = mip_model
                cut_solution = {
                    "max_subcircuit_width": max_subcircuit_width,
                    "subcircuits": subcircuits,
                    "complete_path_map": complete_path_map,
                    "num_cuts": len(positions),
                    "counter": counter,
                    "classical_cost": classical_cost,
                }
    if verbose and len(cut_solution) > 0:
        print("-" * 20)
        classical_cost: float = float(cut_solution["classical_cost"])
        counter = cast(Dict[int, Dict[str, int]], cut_solution["counter"])
        subcircuits: Sequence[Any] = cut_solution["subcircuits"]
        num_cuts: int = cut_solution["num_cuts"]
        print_cutter_result(
            num_subcircuit=len(subcircuits),
            num_cuts=num_cuts,
            subcircuits=subcircuits,
            counter=counter,
            classical_cost=classical_cost,
        )

        if best_mip_model is None:
            raise ValueError(
                "Something went wrong during cut finding. The best MIP model object was never instantiated."
            )
        print("Model objective value = %.2e" % (best_mip_model.objective), flush=True)  # type: ignore
        print("MIP runtime:", best_mip_model.runtime, flush=True)

        if best_mip_model.optimal:
            print("OPTIMAL, MIP gap =", best_mip_model.mip_gap, flush=True)
        else:
            print("NOT OPTIMAL, MIP gap =", best_mip_model.mip_gap, flush=True)
        print("-" * 20, flush=True)
    return cut_solution


def cut_circuit_wire(
    circuit: QuantumCircuit, subcircuit_vertices: Sequence[Sequence[int]], verbose: bool
) -> Dict[str, Any]:
    """
    Perform the provided cuts.

    Used when cut locations are chosen manually.

    Args:
        - circuit (QuantumCircuit): original quantum circuit to be cut into subcircuits
        - subcircuit_vertices (list): the list of vertices to apply the cuts to
        - verbose (bool): whether to print the details of cutting or not

    Returns:
        - (dict): the solution calculated from the provided cuts
    """
    stripped_circ = circuit_stripping(circuit=circuit)
    n_vertices, edges, vertex_ids, id_vertices = read_circuit(circuit=stripped_circ)

    subcircuit_list = []
    for vertices in subcircuit_vertices:
        subcircuit = []
        for vertex in vertices:
            subcircuit.append(id_vertices[vertex])
        subcircuit_list.append(subcircuit)
    if sum([len(subcircuit) for subcircuit in subcircuit_list]) != n_vertices:
        raise ValueError("Not all gates are assigned into subcircuits")

    subcircuit_object = subcircuits_parser(
        subcircuit_gates=subcircuit_list, circuit=circuit
    )
    if len(subcircuit_object) != 2:
        raise ValueError("subcircuit_object should contain exactly two elements.")
    subcircuits = subcircuit_object[0]
    complete_path_map = subcircuit_object[-1]

    O_rho_pairs = get_pairs(complete_path_map=complete_path_map)
    counter = get_counter(subcircuits=subcircuits, O_rho_pairs=O_rho_pairs)
    classical_cost = cost_estimate(counter=counter)
    max_subcircuit_width = max([subcirc.width() for subcirc in subcircuits])  # type: ignore

    cut_solution = {
        "max_subcircuit_width": max_subcircuit_width,
        "subcircuits": subcircuits,
        "complete_path_map": complete_path_map,
        "num_cuts": len(O_rho_pairs),
        "counter": counter,
        "classical_cost": classical_cost,
    }

    if verbose:
        print("-" * 20)
        print_cutter_result(
            num_subcircuit=len(cut_solution["subcircuits"]),
            num_cuts=cut_solution["num_cuts"],
            subcircuits=cut_solution["subcircuits"],
            counter=cut_solution["counter"],
            classical_cost=cut_solution["classical_cost"],
        )
        print("-" * 20)
    return cut_solution


def print_cutter_result(
    num_subcircuit: int,
    num_cuts: int,
    subcircuits: Sequence[QuantumCircuit],
    counter: Dict[int, Dict[str, int]],
    classical_cost: float,
) -> None:
    """
    Pretty print the results.

    Args:
        - num_subciruit (int): the number of subcircuits
        - num_cuts (int): the number of cuts
        - subcircuits (list): the list of subcircuits
        - counter (dict): the dictionary containing all meta information regarding 
            each of the subcircuits
        - classical_cost (float): the estimated processing cost

    Returns:
        - None
    """
    for subcircuit_idx in range(num_subcircuit):
        print("subcircuit %d" % subcircuit_idx)
        print(
            "\u03C1 qubits = %d, O qubits = %d, width = %d, effective = %d, depth = %d, size = %d"
            % (
                counter[subcircuit_idx]["rho"],
                counter[subcircuit_idx]["O"],
                counter[subcircuit_idx]["d"],
                counter[subcircuit_idx]["effective"],
                counter[subcircuit_idx]["depth"],
                counter[subcircuit_idx]["size"],
            )
        )
        print(subcircuits[subcircuit_idx])
    print("Estimated cost = %.3e" % classical_cost, flush=True)
