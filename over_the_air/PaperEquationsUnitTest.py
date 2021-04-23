import pytest
import torch

from over_the_air.paper_equations import (firstTemporalDerivative, secondTemporalDerivative, adversarialLoss,
    thicknessRegularization, roughnessRegularization)


# $ pytest -k DerivativeTest

# Derivative Tests
class DerivativeTest:
    OnesInput = torch.ones(4, 3, 3, 3)
    ZeroesInput = torch.zeros(4, 3, 3, 3)

    def test_firstTemporalDerivative(self):
        # Dimension Check
        assert firstTemporalDerivative(self.OnesInput).size() == torch.Size([4, 3, 3, 3])
        assert firstTemporalDerivative(self.ZeroesInput).size() == torch.Size([4, 3, 3, 3])
        # Output Check
        # Should Output 4 * 3 * 3 * 3 tensor of zeroes
        assert firstTemporalDerivative(self.OnesInput) == self.ZeroesInput

    def test_secondTemporalDerivative(self):
        assert secondTemporalDerivative(self.OnesInput).size() == torch.Size([4, 3, 3, 3])
        assert secondTemporalDerivative(self.ZeroesInput).size() == torch.Size([4, 3, 3, 3])
        # Output Check
        # Should Output 4 * 3 * 3 * 3 tensor of zeroes
        assert secondTemporalDerivative(self.ZeroesInput) == self.ZeroesInput

# Regularization Tests
class RegularizationTest:
    OnesInput = torch.ones(4, 3, 3, 3)
    ZeroesInput = torch.zeros(4, 3, 3, 3)

    def test_thicknessRegularization(self):
        #Dimension Check
        assert thicknessRegularization(self.OnesInput, 1).size() == torch.Size([4, 3, 3, 3])
        assert thicknessRegularization(self.ZeroesInput, 1).size() == torch.Size([4, 3, 3, 3])
        # Output Check
        assert thicknessRegularization(self.OnesInput, 1) == self.OnesInput
        assert thicknessRegularization(self.ZeroesInput, 1) == self.ZeroesInput

    def test_roughnessRegularizaiton(self):
        #Dimension Check
        assert roughnessRegularization(self.OnesInput, 1).size() == torch.Size([4, 3, 3, 3])
        assert roughnessRegularization(self.ZeroesInput, 1).size() == torch.Size([4, 3, 3, 3])
        # Output Check
        assert roughnessRegularization(self.ZeroesInput, 1) == self.ZeroesInput



class TestAdversarialLoss:
    # Labels dimension: same first dimension as predictions
    # second dimension n*1
    # each entry is an entry between 1 and x
    # Predictions
    Pred = torch.eye(10)
    Label = torch.arange(10)

    def test_adversarialLoss(self):
        print(self.Label.shape)
        loss = adversarialLoss(self.Pred, self.Label, 1)
        print(loss)
        assert torch.norm(loss, 1).item() == 20.





