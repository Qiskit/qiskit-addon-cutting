# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Code for converting types of distributions.

.. currentmodule:: circuit_knitting.utils.conversion

.. autosummary::
   :toctree: ../stubs/

   quasi_to_real
   nearest_probability_distribution
   naive_probability_distribution
   dict_to_array
"""

import numpy as np
from qiskit.utils.deprecation import deprecate_func


@deprecate_func(
    removal_timeline="Circuit knitting toolbox 0.8.0 release",
    since="0.7.0",
    package_name="utils",
)
def quasi_to_real(quasiprobability, mode):
    """
    Convert a quasi probability to a valid probability distribution.

    Args:
        quasiprobability: The array of quasiprobabilities
        mode: How to compute the new distribution, either 'nearest' or 'naive'

    Returns:
        The converted probability distribution
    """
    if mode == "nearest":
        return nearest_probability_distribution(quasiprobability=quasiprobability)
    elif mode == "naive":
        return naive_probability_distribution(quasiprobability=quasiprobability)
    else:
        raise NotImplementedError("%s conversion is not implemented" % mode)


@deprecate_func(
    removal_timeline="Circuit knitting toolbox 0.8.0 release",
    since="0.7.0",
    package_name="utils",
)
def nearest_probability_distribution(quasiprobability):
    """
    Convert quasiprobability distribution to the nearest probability distribution.

    Takes a quasiprobability distribution and maps
    it to the closest probability distribution as defined by
    the L2-norm.

    Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).

    Args:
        quasiprobability: The input quasiprobabilities

    Returns:
        The converted probability distribution
    """
    sorted_probs, states = zip(
        *sorted(zip(quasiprobability, range(len(quasiprobability))))
    )
    num_elems = len(sorted_probs)
    new_probs = np.zeros(num_elems)
    beta = 0
    diff = 0
    for state, prob in zip(states, sorted_probs):
        temp = prob + beta / num_elems
        if temp < 0:
            beta += prob
            num_elems -= 1
            diff += prob * prob
        else:
            diff += (beta / num_elems) * (beta / num_elems)
            new_probs[state] = prob + beta / num_elems
    return new_probs


@deprecate_func(
    removal_timeline="Circuit knitting toolbox 0.8.0 release",
    since="0.7.0",
    package_name="utils",
)
def naive_probability_distribution(quasiprobability):
    """
    Convert quasiprobability dist to probability dist by zero-ing out negative values.

    Takes a quasiprobability distribution and does the following two steps:
    1. Update all negative probabilities to 0
    2. Normalize

    Args:
        quasiprobability: The input quasiprobabilities

    Returns:
        The converted probability distribution
    """
    new_probs = np.where(quasiprobability < 0, 0, quasiprobability)
    new_probs /= np.sum(new_probs)
    return new_probs


@deprecate_func(
    removal_timeline="Circuit knitting toolbox 0.8.0 release",
    since="0.7.0",
    package_name="utils",
)
def dict_to_array(distribution_dict, force_prob):
    """
    Convert dictionary of shot results to array of distribution.

    Args:
        distribution_dict: The dictionary containing the shot information
            from circuit execution
        force_prob: Whether to force the distribution to be normalized

    Returns:
        The resulting probability information
    """
    state = list(distribution_dict.keys())[0]
    num_qubits = len(state)
    num_shots = sum(distribution_dict.values())
    cnts = np.zeros(2**num_qubits, dtype=float)
    for state in distribution_dict:
        cnts[int(state, 2)] = distribution_dict[state]
    if abs(sum(cnts) - num_shots) > 1:
        print(
            "dict_to_array may be wrong, converted counts = {}, input counts = {}".format(
                sum(cnts), num_shots
            )
        )
    if not force_prob:
        return cnts
    else:
        prob = cnts / num_shots
        assert abs(sum(prob) - 1) < 1e-10
        return prob
