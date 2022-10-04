from typing import Sequence, Optional, Dict, Callable, Any, Tuple, cast, List

from qiskit import QuantumCircuit
from nptyping import NDArray

from qiskit_ibm_runtime import Sampler

from .wire_cutting import find_wire_cuts, cut_circuit_wire
from .wire_cutting_evaluation import run_subcircuit_instances
from .wire_cutting_post_processing import generate_summation_terms, build
from .wire_cutting_verification import verify, generate_reconstructed_output


class WireCutter:
    def __init__(self, circuit: QuantumCircuit, sampler: Sampler):
        self._circuit = circuit
        self._sampler = sampler

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def sampler(self) -> Sampler:
        return self._sampler

    def cut_automatic(
        self,
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
            circuit=self.circuit,
            max_subcircuit_width=max_subcircuit_width,
            max_cuts=max_cuts,
            num_subcircuits=num_subcircuits,
            max_subcircuit_cuts=max_subcircuit_cuts,
            max_subcircuit_size=max_subcircuit_size,
            verbose=True,
        )

        return cuts

    def cut_manual(
        self, subcircuit_vertices: Sequence[Sequence[int]]
    ) -> Dict[str, Any]:
        """
        Cut the given circuits at the wires specified

        Args:
        """
        if self.circuit is None:
            raise ValueError(
                "A circuit must be passed to the cutter prior to calling a cut method."
            )
        cuts = cut_circuit_wire(
            circuit=self.circuit,
            subcircuit_vertices=subcircuit_vertices,
            verbose=True,
        )

        return cuts

    def evaluate(self, cuts: Dict[str, Any]) -> Dict[int, Dict[int, NDArray]]:
        """
        cuts: results from cutting routine
        """
        _, _, subcircuit_instances = _generate_metadata(cuts)

        subcircuit_instance_probs = _run_subcircuits(
            cuts, subcircuit_instances, self.sampler
        )

        return subcircuit_instance_probs

    def recompose(
        self,
        subcircuit_instance_probs: Dict[int, Dict[int, NDArray]],
        cuts: Dict[str, Any],
        num_threads: int = 1,
    ) -> NDArray:
        summation_terms, subcircuit_entries, _ = _generate_metadata(cuts)

        subcircuit_entry_probs = _attribute_shots(
            subcircuit_entries, subcircuit_instance_probs
        )
        unordered_probability, smart_order = _build(
            cuts, summation_terms, subcircuit_entry_probs, num_threads
        )

        ordered_probability = generate_reconstructed_output(
            self.circuit,
            cuts["subcircuits"],
            unordered_probability,
            smart_order,
            cuts["complete_path_map"],
        )

        return ordered_probability

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
