# TODO: ADD THE PAPER LINK
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import torch
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, is_probability

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE


class OverTheAir(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "delta",
        "regularization_param",
        "beta_1",
        "beta_2"
        "m"
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(self,
                 classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
                 delta: torch.Tensor,
                 regularization_param: float,
                 beta_1: float,
                 beta_2: float,
                 m: float):
        super(OverTheAir, self).__init__(estimator=classifier)

        self.delta = delta
        self.regularization_param = regularization_param
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = m
        self._check_params()

    def generate(self,
                 X: torch.Tensor,
                 labels: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        raise NotImplementedError("")

    @staticmethod
    def firstTemporalDerivative(X: torch.Tensor) -> torch.Tensor:
        """
        Equation 7 from the paper.
        :param X: `torch.tensor`
            Input tensor. Can be any dimensions, but per the paper it should be
            a 4-dimensional Tensor with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        :return: `torch.Tensor`
            The first order temporal derivative with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        """
        # Use dims to ensure that it is only shifted on the first dimensions.
        # Per the paper, we roll x_1,...,x_T in X. Since T is the first
        # dimension of X, we use dim=0.
        return torch.roll(X, 1, dims=0) - torch.roll(X, 0, dims=0)

    @staticmethod
    def secondTemporalDerivative(X: torch.Tensor) -> torch.Tensor:
        """
        Equation 8 from the paper. Defined as:
            Roll(X,-1) - 2*Roll(X, 0) + Roll(X,1)
        :param X: `torch.tensor`
            Input tensor. Can be any dimensions, but per the paper it should be
            a 4-dimensional Tensor with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        :return: `torch.Tensor`
            The first order temporal derivative with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        """
        # Use dims to ensure that it is only shifted on the first dimensions.
        # Per the paper, we roll x_1,...,x_T in X. Since T is the first
        # dimension of X, we use dim=0.
        return (
                torch.roll(X, -1, dims=0)
                - 2 * torch.roll(X, 0, dims=0)
                - torch.roll(X, 1, dims=0)
        )

    def roughnessRegularization(self, delta: torch.Tensor, T: int) -> torch.Tensor:
        """
        ROUGH AND ROWDY
        :param delta: `torch.Tensor`
            Delta parameter from the paper
        :param T:
        :return:
            Rough.
        """
        return 1 / (3 * T) * (
                torch.pow(torch.norm(self.firstTemporalDerivative(delta), 2), 2)
                + torch.pow(torch.norm(self.secondTemporalDerivative(delta), 2), 2)
        )

    # TODO: Also, get rid of the garbage I call most of these comments.
    @staticmethod
    def thicknessRegularization(delta: torch.Tensor, T: int) -> torch.Tensor:
        """
        Thickness Function
        :param delta: `torch.Tensor`
            Delta parameter from the paper
        :param T: `int`

        :return: `torch.Tensor`
            The THICKness. Like oatmeal * oatmeal=oatmeal^2
        """
        return torch.pow(torch.norm(delta, 2), 2) / (3 * T)
