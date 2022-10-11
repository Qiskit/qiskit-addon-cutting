"""File containing the WireCutter class."""
from typing import Sequence, Optional, Dict, Any, Tuple, cast, List, Union

from nptyping import NDArray

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler as TestSampler
from qiskit_ibm_runtime import (
    Sampler,
    Options,
    RuntimeOptions,
    QiskitRuntimeService,
    Session,
)
from quantum_serverless import run_qiskit_remote, get

from .wire_cutting import find_wire_cuts, cut_circuit_wire
from .wire_cutting_evaluation import run_subcircuit_instances
from .wire_cutting_post_processing import generate_summation_terms, build
from .wire_cutting_verification import verify, generate_reconstructed_output


class WireCutter:
    """Class to hold the main cutting functions.

    This is the main class of user interaction for Wire Cutting. This class
    manages all the key user functions such as creating, evaluating, and
    verifying the cuts.

    Attributes:
        - _circuit (QuantumCircuit): original quantum circuit to be
            cut into subcircuits
        - service (QiskitRuntimeService): the runtime service used to simulate or
            execute the subcircuits on real systems
        - _options (Options): the options for the qiskit runtime primitives
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        service: Optional[Union[QiskitRuntimeService, Dict[str, Any]]] = None,
        options: Optional[Options] = None,
        backend_names: Optional[Sequence[str]] = None,
    ):
        """
        Assign the necessary member variable.

        Args:
            - circuit (QuantumCircuit): original quantum circuit to be
                cut into subcircuits
            - service (QiskitRuntimeService): the runtime service used for
                executing the subcircuits
            - options (Options): options for executing in the session
            - backend_names (list): the backends to execute the subcircuits 
                on

        Returns:
            - None
        """
        # Set class fields
        self._circuit = circuit
        self.service = service
        self._options = options
        self._backend_names = backend_names

    @property
    def circuit(self) -> QuantumCircuit: # noqa: D102
        return self._circuit

    @property
    def service(self) -> Optional[QiskitRuntimeService]: # noqa: D102
        return QiskitRuntimeService(**self._service)

    @service.setter
    def service(self, service: Optional[QiskitRuntimeService]) -> None: # noqa: D102
        self._service = service.active_account() if service is not None else service

    @property
    def options(self) -> Optional[Options]: # noqa: D102
        return self._options

    @options.setter
    def options(self, options: Optional[Options]) -> None: # noqa: D102
        self._options = options

    @property
    def backend_names(self) -> Optional[Sequence[str]]: # noqa: D102
        return self._backend_names

    @backend_names.setter
    def backend_names(self, backend_names: Optional[Sequence[str]]) -> None: # noqa: D102
        self._backend_names = backend_names

    def decompose(
        self,
        method: str,
        subcircuit_vertices: Optional[Sequence[Sequence[int]]] = None,
        max_subcircuit_width: Optional[int] = None,
        max_subcircuit_cuts: Optional[int] = None,
        max_subcircuit_size: Optional[int] = None,
        max_cuts: Optional[int] = None,
        num_subcircuits: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Decompose the circuit into a collection of subcircuits.

        Args:
            - method (str): whether to have the cuts be 'automatic'ally found, in a 
                provably optimal way, or whether to 'manual'ly provide the cuts
            - subcircuit_vertices (Sequence[Sequence[int]]): the vertices to be used in 
                the subcircuits. Note that these are not the indices of the qubits, but
                the nodes in the circuit DAG
            - max_subcircuit_width (int): max number of qubits in each subcircuit
            - max_cuts (int): max total number of cuts allowed
            - num_subcircuits (Sequence[int]): list of number of subcircuits to try
            - max_subcircuit_cuts (int, optional): max number of cuts for a subcircuit
            - max_subcircuit_size (int, optional): max number of gates in a subcircuit

        Returns:
            - (Dict[str, Any]): A dictionary containing information on the cuts,
                including the subcircuits themselves (key: 'subcircuits')

        Raises:
            - ValueError: if the input method does not match the other provided arguments
        """
        cuts = {}
        if method == "automatic":
            if max_subcircuit_width is None:
                raise ValueError(
                    "The max_subcircuit_width argument must be set if using automatic cut finding."
                )
            cuts_futures = _cut_automatic(
                self.circuit,
                max_subcircuit_width,
                max_subcircuit_cuts=max_subcircuit_cuts,
                max_subcircuit_size=max_subcircuit_size,
                max_cuts=max_cuts,
                num_subcircuits=num_subcircuits,
            )
            cuts = get(cuts_futures)
        elif method == "manual":
            if subcircuit_vertices is None:
                raise ValueError(
                    "The subcircuit_vertices argument must be set if manually specifying cuts."
                )
            cuts_futures = _cut_manual(self.circuit, subcircuit_vertices)
            cuts = get(cuts_futures)
        else:
            ValueError(
                'The method argument for the decompose method should be either "automatic" or "manual".'
            )
        return cuts

    def evaluate(self, cuts: Dict[str, Any]) -> Dict[int, Dict[int, NDArray]]:
        """
        Evaluate the subcircuits.

        Args:
            - cuts (Dict): the results of cutting
        
        Returns:
            - (Dict): the dictionary containing the results from running
                each of the subcircuits
        """
        _, _, subcircuit_instances = _generate_metadata(cuts)

        subcircuit_instance_probabilities = _run_subcircuits(
            cuts,
            subcircuit_instances,
            self._service,
            self._backend_names,
            self._options,
        )

        return subcircuit_instance_probabilities

    def recompose(
        self,
        subcircuit_instance_probabilities: Dict[int, Dict[int, NDArray]],
        cuts: Dict[str, Any],
        num_threads: int = 1,
    ) -> NDArray:
        """
        Reconstruct the full probabilities from the subcircuit executions.

        Args:
            - subcircuit_instance_probabilities (dict): the probability vectors from each 
                of the subcircuit instances, as output by the _run_subcircuits function
            - cuts (dict): the cuts as found or provided
            - num_threads (int): number of threads to use to parallelize this operation

        Returns:
            - (NDArray): the reconstructed probability vector
        
        """
        reconstructed_probability_futures = _recompose(
            circuit=self.circuit,
            subcircuit_instance_probabilities=subcircuit_instance_probabilities,
            cuts=cuts,
            num_threads=num_threads,
        )
        reconstructed_probabilities = get(reconstructed_probability_futures)

        return reconstructed_probabilities

    def verify(
        self,
        reconstructed_probability: NDArray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare the reconstructed probabilites to the ground truth.

        Note: this function requires the statevector exeuction of the original circuit,
        making it infeasible for most circuits >25 qubits.

        Args:
            - reconstructed_probability (NDArray): the econstructed probabilities
                of the original circuit, obtained from the evaluate method

        Returns:
            - (Dict): a dictionary containing a variety of metrics comparing the
                reconstructed probabilities to the ground truth.
        """
        metrics = verify(
            self.circuit,
            reconstructed_probability,
        )
        return metrics


def _generate_metadata(
    cuts: Dict[str, Any]
) -> Tuple[
    List[Dict[int, int]],
    Dict[int, Dict[Tuple[str, str], Tuple[int, Sequence[Tuple[int, int]]]]],
    Dict[int, Dict[Tuple[Tuple[str, ...], Tuple[Any, ...]], int]],
]:
    """
    Generate metadata used to execute subcircuits and reconstruct probabilities of original circuit.
    
    Args:
        - cuts (Dict[str, Any]): results from the cutting step

    Returns:
        - (tuple): information about the 4^(num cuts) summation terms used to reconstruct original
            probabilities, a dictionary with information on each of the subcircuits, and a dictionary
            containing indexes for each of the subcircuits
    """
    (
        summation_terms,
        subcircuit_entries,
        subcircuit_instances,
    ) = generate_summation_terms(
        subcircuits=cuts["subcircuits"],
        complete_path_map=cuts["complete_path_map"],
        num_cuts=cuts["num_cuts"],
    )
    return summation_terms, subcircuit_entries, subcircuit_instances


def _run_subcircuits(
    cuts: Dict[str, Any],
    subcircuit_instances: Dict[int, Dict[Tuple[Tuple[str, ...], Tuple[Any, ...]], int]],
    service_args: Optional[Dict[str, Any]],
    backend_names: Optional[Sequence[str]],
    options: Optional[Union[Dict, Options]] = None,
) -> Dict[int, Dict[int, NDArray]]:
    """
    Execute all the subcircuit instances.

    task['subcircuit_instance_probs'][subcircuit_idx][subcircuit_instance_idx] = measured prob

    Args:
        - cuts (Dict[str, Any]): results from the cutting step
        - subcircuit_instances (Dict): the dictionary containing the index information for each
            of the subcircuit instances
        - service_args (Dict): the arguments for the runtime service
        - backend_name (str): the method by which the subcircuits should be run
        - options (Options): options for the runtime execution of subcircuits

    Returns:
        - (Dict): the resulting probabilities from each of the subcircuit instances
    """
    subcircuit_instance_probs = run_subcircuit_instances(
        subcircuits=cuts["subcircuits"],
        subcircuit_instances=subcircuit_instances,
        service_args=service_args,
        backend_names=backend_names,
        options=options,
    )

    return subcircuit_instance_probs


def _attribute_shots(
    subcircuit_entries: Dict[
        int, Dict[Tuple[str, str], Tuple[int, Sequence[Tuple[int, int]]]]
    ],
    subcircuit_instance_probs: Dict[int, Dict[int, NDArray]],
) -> Dict[int, Dict[int, NDArray]]:
    """
    Attribute the shots into respective subcircuit entries.

    task['subcircuit_entry_probs'][subcircuit_idx][subcircuit_entry_idx] = prob

    Args:
        - subcircuit_entries (Dict): dictionary containing information about each of the
            subcircuit instances
        - subcircuit_instance_probs (Dict): the probability vectors from each of the subcircuit
            instances, as output by the _run_subcircuits function

    Returns:
        - (Dict): a dictionary containing the probability results to each of the appropriate subcircuits

    Raises:
        - ValueError: if each of the kronecker terms are not of size two or if there are no subcircuit
            probs provided
    """
    subcircuit_entry_probs: Dict[int, Dict[int, NDArray]] = {}
    for subcircuit_idx in subcircuit_entries:
        subcircuit_entry_probs[subcircuit_idx] = {}
        for label in subcircuit_entries[subcircuit_idx]:
            subcircuit_entry_idx, kronecker_term = subcircuit_entries[subcircuit_idx][
                label
            ]
            subcircuit_entry_prob: Optional[NDArray] = None
            for term in kronecker_term:
                if len(term) == 2:
                    coefficient, subcircuit_instance_idx = cast(Tuple[int, int], term)
                else:
                    raise ValueError("Ill-formed Kronecker term: {term}")
                if subcircuit_entry_prob is None:
                    subcircuit_entry_prob = (
                        coefficient
                        * subcircuit_instance_probs[subcircuit_idx][
                            subcircuit_instance_idx
                        ]
                    )
                else:
                    subcircuit_entry_prob += (
                        coefficient
                        * subcircuit_instance_probs[subcircuit_idx][
                            subcircuit_instance_idx
                        ]
                    )

            if subcircuit_entry_prob is None:
                raise ValueError(
                    "Something unexpected happened during shot attribution."
                )
            subcircuit_entry_probs[subcircuit_idx][
                subcircuit_entry_idx
            ] = subcircuit_entry_prob

    return subcircuit_entry_probs


def _build(
    cuts: Dict[str, Any],
    summation_terms: Sequence[Dict[int, int]],
    subcircuit_entry_probs: Dict[int, Dict[int, NDArray]],
    num_threads: int,
) -> Tuple[NDArray, List[int]]:
    """
    Complete the final postprocessing to reconstruct probabilities of the original circuit.
    
    Args:
        - cuts (Dict[str, Any]): results from the cutting step
        - summation_terms (Sequence[Dict[int, int]]): the metadata containing 4^(num cuts) terms
            which represent the final distribution when summed and combined with the subcircuit probabilities
        - subcircuit_entry_probs (Dict[int, Dict[int, NDArray]]): the probabilities for each
            of the subcircuits as calculated from their backend executions
        - mem_limit (int): memory limit for postprocessing
        - num_threads (int): number of threads to use for post processing

    Returns:
        - (tuple): the reconstructed probabilities of the original circuit;
                the first entry contains the unordered probabilities and the second is the order in
                which the subcircuit probabilities should be combined
    """
    reconstructed_prob, smart_order, overhead = build(
        summation_terms=summation_terms,
        subcircuit_entry_probs=subcircuit_entry_probs,
        num_cuts=cuts["num_cuts"],
        num_threads=num_threads,
    )

    unordered_prob = reconstructed_prob
    smart_order = smart_order

    return unordered_prob, smart_order


@run_qiskit_remote()
def _recompose(
    circuit: QuantumCircuit,
    subcircuit_instance_probabilities: Dict[int, Dict[int, NDArray]],
    cuts: Dict[str, Any],
    num_threads: int = 1,
) -> NDArray:
    """
    Reassemble the probability vector using quantum serverless.

    Args:
        - circuit (QuantumCircuit): the original full circuit
        - subcircuit_instance_probabilities (dict): the probability vectors from each 
            of the subcircuit instances, as output by the _run_subcircuits function
        - num_threads (int): the number of threads to use to parallelize the recomposing

    Returns:
        - (NDArray): the reconstructed probability vector
    """
    summation_terms, subcircuit_entries, _ = _generate_metadata(cuts)

    subcircuit_entry_probabilities = _attribute_shots(
        subcircuit_entries, subcircuit_instance_probabilities
    )
    unordered_probability, smart_order = _build(
        cuts, summation_terms, subcircuit_entry_probabilities, num_threads
    )

    reconstructed_probability = generate_reconstructed_output(
        circuit,
        cuts["subcircuits"],
        unordered_probability,
        smart_order,
        cuts["complete_path_map"],
    )

    return reconstructed_probability


@run_qiskit_remote()
def _cut_automatic(
    circuit: QuantumCircuit,
    max_subcircuit_width: int,
    max_subcircuit_cuts: Optional[int] = None,
    max_subcircuit_size: Optional[int] = None,
    max_cuts: Optional[int] = None,
    num_subcircuits: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Automatically find an optimal cut scheme based on input.

    Args:
        - circuit (QuantumCircuit): the circuit to cut
        - max_subcircuit_width (int): max number of qubits in each subcircuit
        - max_cuts (int): max total number of cuts allowed
        - num_subcircuits (Sequence[int]): list of number of subcircuits to try
        - max_subcircuit_cuts (int, optional): max number of cuts for a subcircuit
        - max_subcircuit_size (int, optional): max number of gates in a subcircuit

    Returns:
        - (Dict[str, Any]): A dictionary containing information on the cuts,
            including the subcircuits themselves (key: 'subcircuits')
    
    Rasies:
        - ValueError: if the circuit is not provided (during the initialization 
            of the class)
        - DOcplexException: if the MIP solver is unable to find the optimal solution
    """
    cuts = find_wire_cuts(
        circuit=circuit,
        max_subcircuit_width=max_subcircuit_width,
        max_cuts=max_cuts,
        num_subcircuits=num_subcircuits,
        max_subcircuit_cuts=max_subcircuit_cuts,
        max_subcircuit_size=max_subcircuit_size,
        verbose=True,
    )

    return cuts


@run_qiskit_remote()
def _cut_manual(
    circuit: QuantumCircuit, subcircuit_vertices: Sequence[Sequence[int]]
) -> Dict[str, Any]:
    """
    Cut the given circuits at the wires specified.

    Args:
        - circuit (QuantumCircuit): the circuit to cut
        - subcircuit_vertices (Sequence[Sequence[int]]): the vertices to be used in 
            the subcircuits. Note that these are not the indices of the qubits, but
            the nodes in the circuit DAG.

    Returns:
        - (Dict[str, Any]): A dictionary containing information on the cuts,
            including the subcircuits themselves (key: 'subcircuits')
    """
    cuts = cut_circuit_wire(
        circuit=circuit,
        subcircuit_vertices=subcircuit_vertices,
        verbose=True,
    )

    return cuts
