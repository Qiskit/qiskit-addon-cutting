from typing import Sequence, Optional, Dict, Callable, Any, Tuple, cast, List

from nptyping import NDArray
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import (
    Sampler,
    Options,
    RuntimeOptions,
    QiskitRuntimeService,
    Session,
)
from quantum_serverless import QuantumServerless, run_qiskit_remote, get, put

from .wire_cutting import find_wire_cuts, cut_circuit_wire
from .wire_cutting_evaluation import run_subcircuit_instances
from .wire_cutting_post_processing import generate_summation_terms, build
from .wire_cutting_verification import verify, generate_reconstructed_output


class WireCutter:
    def __init__(
        self,
        circuit: QuantumCircuit,
        service_args: Dict[str, Any],
        options: Optional[Options] = None,
        runtime_options: Optional[RuntimeOptions] = None,
    ):
        # Set class fields
        self._circuit = circuit
        self._service_args = service_args
        self._options = options
        self._runtime_options = runtime_options

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def service_args(self) -> Dict[str, Any]:
        return self._service_args

    @service_args.setter
    def service_args(self, service_args: Dict[str, Any]) -> None:
        self._service_args = service_args

    @property
    def options(self) -> Optional[Options]:
        return self._options

    @options.setter
    def options(self, options: Optional[Options]) -> None:
        self._options = options

    @property
    def runtime_options(self) -> Optional[RuntimeOptions]:
        return self._runtime_options

    @runtime_options.setter
    def runtime_options(self, runtime_options: Optional[RuntimeOptions]) -> None:
        self._runtime_options = runtime_options

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
        cuts: results from cutting routine
        """
        probability_futures = _evaluate(
            cuts, self.service_args, self.options, self.runtime_options
        )
        subcircuit_instance_probabilities = get(probability_futures)

        return subcircuit_instance_probabilities

    def recompose(
        self,
        subcircuit_instance_probabilities: Dict[int, Dict[int, NDArray]],
        cuts: Dict[str, Any],
        num_threads: int = 1,
    ) -> NDArray:
        ordered_probability_futures = _recompose(
            circuit=self.circuit,
            subcircuit_instance_probabilities=subcircuit_instance_probabilities,
            cuts=cuts,
            num_threads=num_threads,
        )
        ordered_probabilities = get(ordered_probability_futures)

        return ordered_probabilities

    def verify(
        self,
        ordered_probability: NDArray,
    ) -> Dict[str, Dict[str, float]]:
        metrics = verify(
            self.circuit,
            ordered_probability,
        )
        return metrics


def _generate_metadata(
    cuts: Dict[str, Any]
) -> Tuple[
    List[Dict[int, int]],
    Dict[int, Dict[Tuple[str, str], Tuple[int, Sequence[Tuple[int, int]]]]],
    Dict[int, Dict[Tuple[Tuple[str, ...], Tuple[Any, ...]], int]],
]:
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
    sampler: Sampler,
) -> Dict[int, Dict[int, NDArray]]:
    """
    Run all the subcircuit instances
    task['subcircuit_instance_probs'][subcircuit_idx][subcircuit_instance_idx] = measured prob
    """
    subcircuit_instance_probs = run_subcircuit_instances(
        subcircuits=cuts["subcircuits"],
        subcircuit_instances=subcircuit_instances,
        sampler=sampler,
    )

    return subcircuit_instance_probs


def _attribute_shots(
    subcircuit_entries: Dict[
        int, Dict[Tuple[str, str], Tuple[int, Sequence[Tuple[int, int]]]]
    ],
    subcircuit_instance_probs: Dict[int, Dict[int, NDArray]],
) -> Dict[int, Dict[int, NDArray]]:
    """
    Attribute the shots into respective subcircuit entries
    task['subcircuit_entry_probs'][subcircuit_idx][subcircuit_entry_idx] = prob
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
def _evaluate(
    cuts: Dict[str, Any],
    service_args: Dict[str, Any],
    options: Optional[Options] = None,
    runtime_options: Optional[RuntimeOptions] = None,
) -> Dict[int, Dict[int, NDArray]]:
    """
    cuts: results from cutting routine
    """
    # Set the backend. Default to runtime qasm simulator
    if (runtime_options is None) or (runtime_options.backend_name is None):
        backend_name = "ibmq_qasm_simulator"
    else:
        backend_name = runtime_options.backend_name

    # Set up our service, session, and sampler primitive
    service = QiskitRuntimeService(**service_args)
    session = Session(service=service, backend=backend_name)
    sampler = Sampler(session=session, options=options)

    _, _, subcircuit_instances = _generate_metadata(cuts)

    subcircuit_instance_probs = _run_subcircuits(cuts, subcircuit_instances, sampler)

    return subcircuit_instance_probs


@run_qiskit_remote()
def _recompose(
    circuit: QuantumCircuit,
    subcircuit_instance_probabilities: Dict[int, Dict[int, NDArray]],
    cuts: Dict[str, Any],
    num_threads: int = 1,
) -> NDArray:
    summation_terms, subcircuit_entries, _ = _generate_metadata(cuts)

    subcircuit_entry_probabilities = _attribute_shots(
        subcircuit_entries, subcircuit_instance_probabilities
    )
    unordered_probability, smart_order = _build(
        cuts, summation_terms, subcircuit_entry_probabilities, num_threads
    )

    ordered_probability = generate_reconstructed_output(
        circuit,
        cuts["subcircuits"],
        unordered_probability,
        smart_order,
        cuts["complete_path_map"],
    )

    return ordered_probability


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
    Automatically find an optimized cut scheme based on input.

    Args:
    max_subcircuit_width: max number of qubits in each subcircuit
    max_cuts: max total number of cuts allowed
    max_subcircuit_cuts: max number of cuts for a subcircuit
    max_subcircuit_size: max number of gates in a subcircuit
    num_subcircuits: list of subcircuits to try
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
    Cut the given circuits at the wires specified

    Args:
    """
    cuts = cut_circuit_wire(
        circuit=circuit,
        subcircuit_vertices=subcircuit_vertices,
        verbose=True,
    )

    return cuts
