# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" op converter """

# pylint: disable=cyclic-import

import logging
from typing import Union, Callable, cast

from qiskit.algorithms import AlgorithmError

from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator
from .weighted_pauli_operator import WeightedPauliOperator

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name,no-else-return,no-member
def to_tpb_grouped_weighted_pauli_operator(
    operator: Union[WeightedPauliOperator, TPBGroupedWeightedPauliOperator],
    grouping_func: Callable,
    **kwargs: int,
) -> TPBGroupedWeightedPauliOperator:
    """

    Args:
        operator: one of supported operator type
        grouping_func: a callable function that grouped the paulis in the operator.
        kwargs: other setting for `grouping_func` function

    Returns:
        the converted tensor-product-basis grouped weighted pauli operator

    Raises:
        AlgorithmError: Unsupported type to convert
    """
    if operator.__class__ == WeightedPauliOperator:
        return grouping_func(operator, **kwargs)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # different tpb grouping approach is asked
        op_tpb = cast(TPBGroupedWeightedPauliOperator, operator)
        if grouping_func != op_tpb.grouping_func and kwargs != op_tpb.kwargs:
            return grouping_func(op_tpb, **kwargs)
        else:
            return op_tpb
    else:
        raise AlgorithmError(
            "Unsupported type to convert to TPBGroupedWeightedPauliOperator: "
            f"{operator.__class__}"
        )
