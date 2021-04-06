import pytest
import torch

from over_the_air.paper_equations import tensorNorm, firstTemporalDerivative, secondTemporalDerivative


# $ pytest -k TestClassDemoInstance -q

class TestPaperEquations:
    testInput = [torch.Tensor([10, 256, 256, 1]),
                 torch.Tensor([10, 256, 256, 1]),
                 torch.Tensor([10, 256, 256, 1]),
                 torch.Tensor([10, 256, 256, 1]),
                 torch.Tensor([10, 256, 256, 1]),
                 torch.Tensor([10, 256, 256, 1])]

    def test_tensorNorm(self):
        for x in TestPaperEquations.testInput:
            assert tensorNorm(x, 1).size() == [1, 4]

        for x in TestPaperEquations.testInput:
            assert tensorNorm(x, 2).size() == [1, 4]

    def test_firstTemporalDerivative(self):
        for x in TestPaperEquations.testInput:
            assert firstTemporalDerivative(x).size() == [1, 4]

    def test_secondTemporalDerivative(self):
        for x in TestPaperEquations.testInput:
            assert secondTemporalDerivative(x).size() == [1, 4]
