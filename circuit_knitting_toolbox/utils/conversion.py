# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np


def quasi_to_real(quasiprobability, mode):
    """
    Convert a quasi probability to a valid probability distribution
    """
    if mode == "nearest":
        return nearest_probability_distribution(quasiprobability=quasiprobability)
    elif mode == "naive":
        return naive_probability_distribution(quasiprobability=quasiprobability)
    else:
        raise NotImplementedError("%s conversion is not implemented" % mode)


def nearest_probability_distribution(quasiprobability):
    """Takes a quasiprobability distribution and maps
    it to the closest probability distribution as defined by
    the L2-norm.
    Parameters:
        return_distance (bool): Return the L2 distance between distributions.
    Returns:
        ProbDistribution: Nearest probability distribution.
        float: Euclidean (L2) distance of distributions.
    Notes:
        Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
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


def naive_probability_distribution(quasiprobability):
    """
    Takes a quasiprobability distribution and does the following two steps:
    1. Update all negative probabilities to 0
    2. Normalize
    """
    new_probs = np.where(quasiprobability < 0, 0, quasiprobability)
    new_probs /= np.sum(new_probs)
    return new_probs


def dict_to_array(distribution_dict, force_prob):
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
