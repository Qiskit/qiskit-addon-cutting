"""File containing the class used for specifying characteristics of the target QPU."""


class DeviceConstraints:

    """Class for specifying the characteristics of the target quantum
    processor that the optimizer must respect in order for the resulting
    subcircuits to be executable on the target processor.

    Member Variables:

    qubits_per_QPU (int) : The number of qubits that are available on the
    individual QPUs that make up the quantum processor.

    num_QPUs (int) : The number of QPUs in the target quantum processor.

    Raises:

    ValueError: qubits_per_QPU must be a positive integer.
    ValueError: num_QPUs must be a positive integer.
    """

    def __init__(self, qubits_per_QPU, num_QPUs):
        if not (isinstance(qubits_per_QPU, int) and qubits_per_QPU > 0):
            raise ValueError("qubits_per_QPU must be a positive definite integer.")

        if not (isinstance(num_QPUs, int) and num_QPUs > 0):
            raise ValueError("num_QPUs must be a positive definite integer.")

        self.qubits_per_QPU = qubits_per_QPU
        self.num_QPUs = num_QPUs

    def getQPUWidth(self):
        """Return the number of qubits supported on each individual QPU."""
        return self.qubits_per_QPU
