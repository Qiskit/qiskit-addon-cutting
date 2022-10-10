"""Functions for comparing array distances."""
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
import copy

import numpy as np


def chi2_distance(target, obs):  # noqa: D301
    r"""
    Measure the Chi-square distance.

    The Chi-Square distance is a measure of statistically correlation between
    two feature vectors and is defined as $ \sum_i \frac{(x_i - y_i)^2}{x_i + y_i}$.

    Examples:
    >>> chi2_distance(np.array([0.1, 0.1, 0.3, 0.5]), np.array([0.25, 0.25, 0.25, 0.25]))
    0.21645021645021645

    >>> chi2_distance(np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.25, 0.25, 0.25, 0.25]))
    0

    Args:
        - target (NDArray): the target feature vector
        - obs (NDArray): the actually observed feature vector

    Returns:
        - (Float): the computed distance

    Raises:
        - Exception: if the target is not a numpy array or dictionary, an exception
            is raised
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    obs = np.absolute(obs)
    if isinstance(target, np.ndarray):
        assert len(target) == len(obs)
        distance = 0
        for t, o in zip(target, obs):
            if abs(t - o) > 1e-10:
                distance += np.power(t - o, 2) / (t + o)
    elif isinstance(target, dict):
        distance = 0
        for o_idx, o in enumerate(obs):
            if o_idx in target:
                t = target[o_idx]
                if abs(t - o) > 1e-10:
                    distance += np.power(t - o, 2) / (t + o)
            else:
                distance += o
    else:
        raise Exception("Illegal target type:", type(target))
    return distance


def MSE(target, obs):  # noqa: D301
    r"""
    Compute the Mean Squared Error (MSE).

    The MSE is a common metric in fields such as deep learning and is used to
    measure the squared distance between two vectors via:
    $\sum_i (x_i - y_i)^2$.

    Example:
    >>> MSE(np.array([0.1, 0.1, 0.3, 0.5]), np.array([0.25, 0.25, 0.25, 0.25]))
    0.0275

    Args:
        - target (NDArray): the target feature vector
        - obs (NDArray): the actually observed feature vector

    Returns:
        - (Float): the computed MSE

    Raises:
        - Exception: if the target is not a dict, or if the target and obs are not
            numpy arrays or if the target is not a numpy array and the obs are not
            a dict an expection is raised
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    if isinstance(target, dict):
        se = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    elif isinstance(target, np.ndarray) and isinstance(obs, np.ndarray):
        target = target.reshape(-1, 1)
        obs = obs.reshape(-1, 1)
        squared_diff = (target - obs) ** 2
        se = np.sum(squared_diff)
        mse = np.mean(squared_diff)
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        se = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    else:
        raise Exception("target type : %s" % type(target))
    return mse


def MAPE(target, obs):  # noqa: D301
    r"""
    Compute the Mean Absolute Percentage Error (MAPE).

    The MAPE is a scaled metric in the range [0, 100] defining the percentage
    difference between two vectors via:
    $ \sum_i \frac{x_i - y_i}{x_i} $.

    Example:
    >>> MAPE(np.array([0.1, 0.1, 0.3, 0.5]), np.array([0.25, 0.25, 0.25, 0.25]))
    91.66666666666659

    Args:
        - target (NDArray): the target feature vector
        - obs (NDArray): the actually observed feature vector

    Returns:
        - (Float): the computed MAPE

    Raises:
        - Exception: if the target is not a dict, or if the target and obs are not
            numpy arrays or if the target is not a numpy array and the obs are not
            a dict an expection is raised
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    epsilon = 1e-16
    if isinstance(target, dict):
        curr_sum = np.sum(list(target.values()))
        new_sum = curr_sum + epsilon * len(target)
        mape = 0
        for t_idx in target:
            t = (target[t_idx] + epsilon) / new_sum
            o = obs[t_idx]
            mape += abs((t - o) / t)
        mape /= len(obs)
    elif isinstance(target, np.ndarray) and isinstance(obs, np.ndarray):
        target = target.flatten()
        target += epsilon
        target /= np.sum(target)
        obs = obs.flatten()
        obs += epsilon
        obs /= np.sum(obs)
        mape = np.abs((target - obs) / target)
        mape = np.mean(mape)
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        curr_sum = np.sum(list(target.values()))
        new_sum = curr_sum + epsilon * len(target)
        mape = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = (target[o_idx] + epsilon) / new_sum
            mape += abs((t - o) / t)
        mape /= len(obs)
    else:
        raise Exception("target type : %s" % type(target))
    return mape * 100


def cross_entropy(target, obs):  # noqa: D301
    """
    Compue the cross entropy between two distributions.

    The cross entropy is a measure of the difference between two probability
    distributions, defined via:
    $ -\sum_i x_i \log y_i $.

    Example:
    >>> cross_entropy(np.array([0.1, 0.1, 0.3, 0.5]), np.array([0.25, 0.25, 0.25, 0.25]))
    1.3862943611198906

    Args:
        - target (NDArray): the target feature vector
        - obs (NDArray): the actually observed feature vector

    Returns:
        - (Float): the computed cross entropy

    Raises:
        - Exception: if the target is not a dict, or if the target and obs are not
            numpy arrays or if the target is not a numpy array and the obs are not
            a dict an expection is raised
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    if isinstance(target, dict):
        CE = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            o = o if o > 1e-16 else 1e-16
            CE += -t * np.log(o)
        return CE
    elif isinstance(target, np.ndarray) and isinstance(obs, np.ndarray):
        obs = np.clip(obs, a_min=1e-16, a_max=None)
        CE = np.sum(-target * np.log(obs))
        return CE
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        CE = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            o = o if o > 1e-16 else 1e-16
            CE += -t * np.log(o)
        return CE
    else:
        raise Exception("target type : %s, obs type : %s" % (type(target), type(obs)))


def HOP(target, obs):
    """
    Compue the Heavy Output Probability (HOP).

    The HOP is an important metric for quantum volume experiments and is defined at the
    probability that one measures a bitstring above the median target probability.

    Example:
    >>> HOP(np.array([0.1, 0.1, 0.3, 0.5]), np.array([0.25, 0.25, 0.25, 0.25]))
    0.5

    Args:
        - target (NDArray): the target feature vector
        - obs (NDArray): the actually observed feature vector

    Returns:
        - (Float): the computed HOP
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    target_median = np.median(target)
    hop = 0
    for t, o in zip(target, obs):
        if t > target_median:
            hop += o
    return hop
