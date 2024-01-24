# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum circuit representation compatible with cut-finding optimizers."""

from __future__ import annotations

import copy
import string
import numpy as np
from typing import NamedTuple
from abc import ABC, abstractmethod




class CircuitElement(NamedTuple):
    """Named tuple for specifying a circuit element."""

    name: str
    params: list 
    qubits: tuple
    gamma: float | int


class CircuitInterface(ABC):

    """Base class for accessing and manipulating external circuit
    representations, and for converting external circuit representations
    to the internal representation used by the circuit cutting optimization code.

    Derived classes must override the default implementations of the abstract
    methods defined in this base class.
    """

    @abstractmethod
    def getNumQubits(self):
        """Derived classes must override this function and return the number
        of qubits in the input circuit."""

    @abstractmethod
    def getMultiQubitGates(self):
        """Derived classes must override this function and return a list that
        specifies the multiqubit gates in the input circuit.

        The returned list is of the form:
            [ ... [<unique_index> <gate_specification> <cut_constaints>] ...]

        The <unique_index> can be any object that uniquely identifies the gate
        in the circuit. The <unique_index> can be used as an argument in other
        member functions implemented by the derived class to replace the gate
        with the decomposition determined by the optimizer.

        The <gate_specification> must of the form
            (<gate_name>, <qubit_id_1>, ..., <qubit_id_n>)

        The <gate_name> must be a hashable identifier that can be used to
        look up cutting rules for the specified gate. Gate names are typically
        the Qiskit names of the gates.

        The <qubit_id> must be a non-negative integer with qubits numbered
        starting with zero.  Derived classes are responsible for constructing the
        mappings from external qubit identifiers to the corresponding qubit IDs.

        The <cut_constaints> can be of the form
            None
            []
            [None]
            [<cut_type_1>, ..., <cut_type_n>]

        A cut constraint of None indicates that no constraints are placed
        on how or whether cuts can be performed. An empty list [] or the
        list [None] indicates that no cuts are to be performed and the gate
        is to be applied without cutting. A list of cut types of the form
        [<cut_type_1> ... <cut_type_n>] indicates precisely which types of
        cuts can be considered. In this case, the cut type None must be
        explicitly included to indicate the possibilty of not cutting, if
        not cutting is to be considered. In the current version of the code,
        the allowed cut types are 'None', 'GateCut', 'WireCut', and 'AbsorbGate'.
        """

    @abstractmethod
    def insertGateCut(self, gate_ID, cut_type):
        """Derived classes must override this function and mark the specified
        gate as being cut.  The cut type can only be "LO" in this release.
        """

    @abstractmethod
    def insertWireCut(self, gate_ID, input_ID, src_wire_ID, dest_wire_ID, cut_type):
        """Derived classes must override this function and insert a wire cut
        into the output circuit just prior to the specified gate on the wire
        connected to the specified input of that gate.  Gate inputs are
        numbered starting from 1.  The wire/qubit ID of the wire to be cut
        is also provided as input to allow the wire choice to be verified.
        The ID of the new wire/qubit is also provided, which can then be used
        internally in derived classes to create new wires/qubits as needed.
        The cut type can only be "LO" in this release.
        """


    @abstractmethod
    def defineSubcircuits(self, list_of_list_of_wires):
        """Derived classes must override this function.  The input is a
        list of subcircuits where each subcircuit is specified as a
        list of wire IDs.
        """



class SimpleGateList(CircuitInterface):

    """Derived class that converts a simple list of gates into
    the form needed by the circuit-cutting optimizer code.

    Elements of the list must be of the form:
        'barrier'
        ('barrier' <qubit_name>)
        (<gate_name> <qubit_name_1> ... <qubit_name_n>)

    Qubit names can be any hashable objects. Gate names can also be any
    hashable objects, but they must be consistent with the names used by the
    optimizer to look up cutting rules for the specified gates.

    The constructor can be supplied with a list of qubit names to force a
    preferred ordering in the assignment of numeric qubit IDs to each name.

    Member Variables:

    qubit_names (NameToIDMap) is an object that maps qubit names to
    numerical qubit IDs.

    num_qubits (int) is the number of qubits in the input circuit. Qubit IDs
    whose values are greater than or equal to num_qubits represent qubits
    that were introduced as the result of wire cutting.  These qubits are
    assigned generated names of the form ('cut', <qubit_name>) in the
    qubit_names object, where <qubit_name> is the name of the wire/qubit
    that was cut to create the new wire/qubit.

    circuit (list) is the internal representation of the circuit, which is
    a list of the following form:

        [ ... [<gate_specification>, None] ...]

    where the qubit names have been replaced with qubit IDs in the gate
    specifications.

    new_circuit (list) is a list of gate specifications that define
    the cut circuit.  As with circuit, qubit IDs are used to identify
    wires/qubits.

    cut_type (list) is a list that assigns cut-type annotations to gates
    in new_circuit to indicate which quasiprobability decomposition to
    use for the corresponding gate/wire cut.

    new_gate_ID_map (list) is a list that maps the positions of gates
    in circuit to their new positions in new_circuit.

    output_wires (list) maps qubit IDs in circuit to the corresponding
    output wires of new_circuit so that observables defined for circuit
    can be remapped to new_circuit.

    subcircuits (list) is a list of list of wire IDs, where each list of
    wire IDs defines a subcircuit.
    """

    def __init__(self, input_circuit, init_qubit_names=[]):
        self.qubit_names = NameToIDMap(init_qubit_names)

        self.circuit = list()
        self.new_circuit = list()
        self.cut_type = list()

        for gate in input_circuit:
            self.cut_type.append(None)
            if not isinstance(gate, CircuitElement):
                assert gate == "barrier"
                self.circuit.append([gate, None])
                self.new_circuit.append(gate),
            else: 
                    gate_spec = CircuitElement(
                    name=gate.name,
                    params = gate.params,
                    qubits=tuple(self.qubit_names.getID(x) for x in gate.qubits),
                    gamma=gate.gamma,
                )
            self.circuit.append([gate_spec, None])
            self.new_circuit.append(gate_spec)

        self.new_gate_ID_map = np.arange(len(self.circuit), dtype=int)
        self.num_qubits = self.qubit_names.getArraySizeNeeded()
        self.output_wires = np.arange(self.num_qubits, dtype=int)

        # Initialize the list of subcircuits assuming no cutting
        self.subcircuits = list(list(range(self.num_qubits)))

        # Initialize the graph of strongly connected subcircuits
        # assuming LO decompositions (i.e., no communication between
        # subcircuits)
        self.scc_subcircuits = [(s,) for s in range(len(self.subcircuits))]
        self.scc_order = np.zeros(
            (len(self.scc_subcircuits), len(self.scc_subcircuits)), dtype=bool
        )



    def getNumQubits(self):
        """Return the number of qubits in the input circuit."""

        return self.num_qubits

    def getNumWires(self):
        """Return the number of wires/qubits in the cut circuit."""

        return self.qubit_names.getNumItems()

    def getMultiQubitGates(self):
        """Extract the multiqubit gates from the circuit and prepends the
        index of the gate in the circuits to the gate specification.

        The elements of the resulting list therefore have the form
            [<index> <gate_specification> <cut_constaints>]

        The <gate_specification> and <cut_constaints> have the forms
        described above.

        The <index> is the list index of the corresponding element in
        self.circuit
        """
        subcircuit = list()
        for k, gate in enumerate(self.circuit):
            if gate[0] != "barrier":
                if len(gate[0].qubits) > 1 and gate[0].name != "barrier":
                    subcircuit.append([k] + gate)

        return subcircuit

    def insertGateCut(self, gate_ID, cut_type):
        """Mark the specified gate as being cut.  The cut type in this release
        can only be "LO".
        """

        gate_pos = self.new_gate_ID_map[gate_ID]
        self.cut_type[gate_pos] = cut_type

    def insertWireCut(self, gate_ID, input_ID, src_wire_ID, dest_wire_ID, cut_type):
        """Insert a wire cut into the output circuit just prior to the
        specified gate on the wire connected to the specified input of
        that gate.  Gate inputs are numbered starting from 1.  The
        wire/qubit ID of the source wire to be cut is also provided as
        input to allow the wire choice to be verified.  The ID of the
        (new) destination wire/qubit must also be provided.  The cut
        type in this release can only be "LO".
        """

        gate_pos = self.new_gate_ID_map[gate_ID]
        new_gate = self.new_circuit[gate_pos]
        new_gate_spec = self.new_circuit[gate_pos]

        assert src_wire_ID == new_gate_spec[input_ID ], (
            f"Input wire ID {src_wire_ID} does not match "
            + f"new_circuit wire ID {new_gate_spec[input_ID]}"
        )

        # If the new wire does not yet exist, then define it
        if self.qubit_names.getName(dest_wire_ID) is None:
            wire_name = self.qubit_names.getName(src_wire_ID)
            self.qubit_names.defineID(dest_wire_ID, ("cut", wire_name))

        # Replace src_wire_ID with dest_wire_ID in the part of new_circuit that
        # follows the wire-cut insertion point
        wire_map = np.arange(self.qubit_names.getArraySizeNeeded(), dtype=int)
        wire_map[src_wire_ID] = dest_wire_ID
        self.replaceWireIDs(self.new_circuit[gate_pos:], wire_map)

        # Insert a move operator
        self.new_circuit.insert(gate_pos, ["move", src_wire_ID, dest_wire_ID])
        self.cut_type.insert(gate_pos, cut_type)
        self.new_gate_ID_map[gate_ID:] += 1

        # Update the output wires
        qubit = self.circuit[gate_ID][0][input_ID]
        self.output_wires[qubit] = dest_wire_ID


    def defineSubcircuits(self, list_of_list_of_wires):
        """Assign subcircuits where each subcircuit is
        specified as a list of wire IDs.
        """

        self.subcircuits = list_of_list_of_wires

    def getWireNames(self):
        """Return a list of the internal wire names used in the circuit,
        which consists of the original qubit names together with additional
        names of form ("cut", <name>) introduced to represent cut wires.
        """

        return list(self.qubit_names.getItems())

    def exportCutCircuit(self, name_mapping="default"):
        """Return a list of gates representing the cut circuit.  If None
        is provided as the name_mapping, then the original qubit names are
        used with additional names of form ("cut", <name>) introduced as
        needed to represent cut wires.  If "default" is used as the mapping
        then the defaultWireNameMapping() method defines the name mapping.
        Otherwise, the name_mapping is assumed to be a dictionary that maps
        internal wire names to desired names.
        """

        wire_map = self.makeWireMapping(name_mapping)
        out = copy.deepcopy(self.new_circuit)

        self.replaceWireIDs(out, wire_map)

        return out

    def exportOutputWires(self, name_mapping="default"):
        """Return a dictionary that maps output qubits in the input circuit
        to the corresponding output wires/qubits in the cut circuit.  If None
        is provided as the name_mapping, then the original qubit names are
        used with additional names of form ("cut", <name>) introduced as
        needed to represent cut wires.  If "default" is used as the mapping
        then the defaultWireNameMapping() method defines the name mapping.
        Otherwise, the name_mapping is assumed to be a dictionary that maps
        internal wire names to desired names.
        """

        wire_map = self.makeWireMapping(name_mapping)
        out = dict()
        for in_wire, out_wire in enumerate(self.output_wires):
            out[self.qubit_names.getName(in_wire)] = wire_map[out_wire]
        return out

    def exportSubcircuitsAsString(self, name_mapping="default"):
        """Return a string that maps qubits/wires in the output circuit
        to subcircuits per the Circuit Knitting Toolbox convention.  This
        method only works with mappings to numeric qubit/wire names, such
        as provided by "default" or a custom name_mapping.
        """

        wire_map = self.makeWireMapping(name_mapping)

        out = list(range(self.getNumWires()))
        alphabet = string.ascii_uppercase + string.ascii_lowercase
        for k, subcircuit in enumerate(self.subcircuits):
            for wire in subcircuit:
                out[wire_map[wire]] = alphabet[k]
    
        return "".join(out)

    def makeWireMapping(self, name_mapping):
        """Return a wire-mapping array given an input specification of a
        name mapping.  If None is provided as the input name_mapping, then
        the original qubit names are mapped to themselves.  If "default"
        is used as the name_mapping, then the defaultWireNameMapping()
        method is used to define the name mapping.  Otherwise, name_mapping
        itself is assumed to be the dictionary to use.
        """

        if name_mapping is None:
            name_mapping = dict()
            for name in self.getWireNames():
                name_mapping[name] = name

        elif name_mapping == "default":
            name_mapping = self.defaultWireNameMapping()

        wire_mapping = [None for x in range(self.qubit_names.getArraySizeNeeded())]

        for k in self.qubit_names.getIDs():
            wire_mapping[k] = name_mapping[self.qubit_names.getName(k)]

        return wire_mapping

    def defaultWireNameMapping(self):
        """Return a dictionary that maps wire names in self.qubit_names to
        default numeric output qubit names when exporting a cut circuit.  Cut
        wires are assigned numeric names that are adjacent to the numeric
        name of the wire prior to cutting so that Move operators are then
        applied against adjacent qubits.
        """

        name_pairs = [(name, self.sortOrder(name)) for name in self.getWireNames()]

        name_pairs.sort(key=lambda x: x[1])

        name_map = dict()
        for k, pair in enumerate(name_pairs):
            name_map[pair[0]] = k

        return name_map

    def sortOrder(self, name):
        if isinstance(name, tuple):
            if name[0] == "cut":
                x = self.sortOrder(name[1])
                x_int = int(x)
                x_frac = x - x_int
                return x_int + 0.5 * x_frac + 0.5

        return self.qubit_names.getID(name)

    def replaceWireIDs(self, gate_list, wire_map):
        """Iterate through a list of gates and replaces wire IDs with the
        values defined by the wire_map.
        """

        for gate in gate_list:
            for k in range(1, len(gate)):
                gate[k] = wire_map[gate[k]]


class NameToIDMap:

    """Class used to map hashable items (e.g., qubit names) to natural numbers
    (e.g., qubit IDs)"""

    def __init__(self, init_names=[]):
        """Allow the name dictionary to be initialized with the names
        in init_names in the order the names appear in order to force a
        preferred ordering in the assigment of item IDs to those names.
        """

        self.next_ID = 0
        self.item_dict = dict()
        self.ID_dict = dict()

        for name in init_names:
            self.getID(name)

    def getID(self, item_name):
        """Return the numeric ID associated with the specified hashable item.
        If the hashable item does not yet appear in the item dictionary, a new
        item ID is assigned.
        """


        if item_name not in self.item_dict:
            while self.next_ID in self.ID_dict:
                self.next_ID += 1

            self.item_dict[item_name] = self.next_ID
            self.ID_dict[self.next_ID] = item_name
            self.next_ID += 1

        return self.item_dict[item_name]

    def defineID(self, item_ID, item_name):
        """Assign a specific ID number to an item name."""

        assert item_ID not in self.ID_dict, f"item ID {item_ID} already assigned"
        assert (
            item_name not in self.item_dict
        ), f"item name {item_name} already assigned"

        self.item_dict[item_name] = item_ID
        self.ID_dict[item_ID] = item_name

    def getName(self, item_ID):
        """Return the name associated with the specified item ID.
        None is returned if item_ID does not (yet) exist.
        """

        if item_ID not in self.ID_dict:
            return None

        return self.ID_dict[item_ID]

    def getNumItems(self):
        """Return the number of hashable items loaded thus far."""

        return len(self.item_dict)

    def getArraySizeNeeded(self):
        """Return one plus the maximum item ID assigned thus far,
        or zero if no items have been assigned.  The value returned
        is thus the minimum size needed to construct a Python/Numpy
        array that maps item IDs to other values.
        """

        if self.getNumItems() <= 0:
            return 0

        return 1 + max(self.ID_dict.keys())

    def getItems(self):
        """Return an iterator over the hashable items loaded thus far."""

        return self.item_dict.keys()

    def getIDs(self):
        """Return an iterator over the hashable items loaded thus far."""

        return self.ID_dict.keys()
