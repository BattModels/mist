import torch
from src.generate import (
    AnyCritic,
    CriticPanel,
    EquationCritic,
    EquationOracle,
    OracleCritic,
    QuadrantCritic,
    limits_to_bounds,
    logit_limits,
)
from torch import nn


class TestUtilityFunctions:
    def test_logit_limits_basic(self):
        channels = ["a", "b", "c"]
        # pass_positive=True but flip channel “b”
        limits_positive = logit_limits(
            channels, pass_positive=True, flip_channels={"b": True}
        )
        # “a” was not flipped and pass_positive=True => (0, None)
        assert limits_positive["a"] == (0, None)
        # “b” was flipped => (None, 0)
        assert limits_positive["b"] == (None, 0)
        # “c” untouched => (0, None)
        assert limits_positive["c"] == (0, None)

    def test_limits_to_bounds_full(self):
        limits = {"a": (0.0, 2.0), "b": (-1.0, 1.0)}
        channels = ["a", "b", "c"]
        lower, upper = limits_to_bounds(limits, channels)
        # a → [0.0, 2.0], b → [−1.0, 1.0], c unconstrained → [−inf, inf]
        expected_lower = [0.0, -1.0, -float("inf")]
        expected_upper = [2.0, 1.0, float("inf")]
        assert torch.allclose(torch.tensor(lower), torch.tensor(expected_lower))
        assert torch.allclose(torch.tensor(upper), torch.tensor(expected_upper))


class TestQuadrantCritic:
    def test_from_all_passing_default(self):
        channels = ["x", "y"]
        critic = QuadrantCritic.from_all_passing(channels)
        # pass_positive=True by default ⇒ both channels get limits (0, None).
        # lower should be [[0.0, 0.0]], upper [[inf, inf]]
        assert torch.all(critic.lower == torch.tensor([[0.0, 0.0]]))
        assert torch.all(critic.upper == torch.tensor([[float("inf"), float("inf")]]))

    def test_from_all_passing_flipped(self):
        channels = ["x", "y"]
        # flip x ⇒ x is (None, 0), y is (0, None)
        critic = QuadrantCritic.from_all_passing(
            channels, pass_positive=True, flip_channels={"x": True}
        )
        assert torch.all(critic.lower == torch.tensor([[-float("inf"), 0.0]]))
        assert torch.all(critic.upper == torch.tensor([[0.0, float("inf")]]))

    def test_active_channels_property(self):
        # Only “a” is constrained; “b” and “c” are unconstrained ⇒ a is “active,” others are not
        limits = {"a": (0.0, 1.0)}
        channels = ["a", "b", "c"]
        critic = QuadrantCritic.from_limits(limits, channels)
        expected = torch.tensor([True, False, False])
        assert torch.all(critic.active_channels == expected)


class TestAnyCritic:
    def test_from_any_passing_basic(self):
        channels = ["p", "q", "r"]
        critic = AnyCritic.from_any_passing(
            channels,
            pass_positive=False,
            flip_channels={"q": False},
            subset=["r"],
        )
        lower_expected, upper_expected = limits_to_bounds(
            logit_limits(channels, pass_positive=False, flip_channels={"q": False}),
            channels,
        )
        mask_expected = torch.tensor([False, False, True])

        assert torch.all(critic.lower == torch.tensor(lower_expected).view(1, -1))
        assert torch.all(critic.upper == torch.tensor(upper_expected).view(1, -1))
        assert torch.all(critic.mask == mask_expected.view(1, -1))

        test_cases = [
            ([-2, 1, -3], True),
            ([-2, 1, 3], False),
            ([-2, -1, 3], False),
            ([2, -1, -3], True),
        ]
        self.check_any_call_logic(critic, test_cases)

    def test_logic(self):
        channels = ["a", "b", "c"]
        critic = AnyCritic.from_any_passing(channels)
        test_cases = [
            ([1, 2, 3], True),
            ([-1, -2, -3], False),
            ([-1, 1, -2], True),
            ([-1, -1, 1], True),
        ]
        self.check_any_call_logic(critic, test_cases)

    def check_any_call_logic(self, critic, test_cases):
        for y, expected in test_cases:
            out = critic(torch.tensor(y))
            assert out.shape == (1,)
            assert out == torch.tensor([expected]), f"y: {y} -> {out} vs. {expected}"

        y = torch.stack([torch.tensor(tc[0]) for tc in test_cases])
        expected = torch.tensor([tc[1] for tc in test_cases])
        assert y.shape == (len(test_cases), 3)
        assert expected.shape == (len(test_cases),)
        out = critic(y)
        assert torch.all(out == expected)

    def test_any_call_logic(self):
        # channel 0: [0.0, 1.0]; channel1: unconstrained; channel2: [2.0, 3.0]
        lower = torch.tensor([0.0, -float("inf"), 2.0])
        upper = torch.tensor([1.0, float("inf"), 3.0])
        mask = torch.tensor([True, False, True])  # only channels 0 and 2 are considered
        critic = AnyCritic(lower, upper, mask)

        test_cases = [
            ([0.5, 100.0, 2.5], True),
            ([1.5, 0.0, 2.5], True),
            ([0.5, 0.0, 3.5], True),
            ([1.5, 0.0, 3.5], False),
            ([1.5, 0.0, 3.5], False),
            ([1.5, 0.0, 1.5], False),
        ]
        self.check_any_call_logic(critic, test_cases)

    def test_active_channels(self):
        lower = torch.tensor([-float("inf"), 0.0])
        upper = torch.tensor([float("inf"), 1.0])
        mask = torch.tensor([True, False])
        critic = AnyCritic(lower, upper, mask)
        expected = torch.tensor([False, False])
        assert torch.all(critic.active_channels == expected)


class DummyOracle(nn.Module):
    """A simple oracle that returns a constant vector and declares its channels."""

    def __init__(self, channels, output):
        super().__init__()
        self.channels = [{"name": name} for name in channels]
        self._output = torch.tensor(output, dtype=torch.float32)

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        return self._output.expand(batch_size, -1)


class TestOracleCriticAndPanel:
    def test_oracle_critic_forward(self):
        # DummyOracle returns [1.0, -1.0]
        oracle = DummyOracle(channels=["a", "b"], output=[1.0, -1.0])

        # 1) Critic that passes when a∈(0,2) and b∈(-2,0) => both True
        limits1 = {"a": (0.0, 2.0), "b": (-2.0, 0.0)}
        critic1 = QuadrantCritic.from_limits(limits1, ["a", "b"])
        oc1 = OracleCritic(oracle, critic1)

        input_ids = torch.ones((3, 5), dtype=torch.int64)
        y_out, score_out = oc1(input_ids)

        # y_out: repeating [1.0, -1.0]
        expected_y = torch.tensor([[1.0, -1.0]]).expand(3, -1)
        assert torch.allclose(y_out, expected_y)
        # score_out: a=1.0 ∈ (0,2) & b=−1.0 ∈ (−2,0) => True
        assert torch.all(score_out == torch.tensor([True, True, True]))

        # 2) Critic that requires b∈(0,2) => b=−1.0 fails => always False
        limits2 = {"a": (0.0, 2.0), "b": (0.0, 2.0)}
        critic2 = QuadrantCritic.from_limits(limits2, ["a", "b"])
        oc2 = OracleCritic(oracle, critic2)
        _, score_out2 = oc2(input_ids)
        assert torch.all(score_out2 == torch.tensor([False, False, False]))

    def test_critic_panel_combination(self):
        # Oracle1 returns [0.5], channel "x". Critic1: x∈(0,1) => True
        oracle1 = DummyOracle(channels=["x"], output=[0.5])
        limits1 = {"x": (0.0, 1.0)}
        critic1 = QuadrantCritic.from_limits(limits1, ["x"])
        oc1 = OracleCritic(oracle1, critic1)

        # Oracle2 returns [-1.0, 2.0], channels ["y","z"]. Critic2: y∈(-2,0) & z∈(1,3)
        oracle2 = DummyOracle(channels=["y", "z"], output=[-1.0, 2.0])
        limits2 = {"y": (-2.0, 0.0), "z": (1.0, 3.0)}
        critic2 = QuadrantCritic.from_limits(limits2, ["y", "z"])
        oc2 = OracleCritic(oracle2, critic2)

        panel = CriticPanel([oc1, oc2])

        # Channels should be ["x", "y", "z"]
        assert panel.channels == ["x", "y", "z"]

        # Batch of size 2
        input_ids = torch.zeros((2, 3), dtype=torch.int64)
        y_combined, net_score = panel(input_ids)

        # y_combined = [[0.5, -1.0, 2.0], [0.5, -1.0, 2.0]]
        expected_y = torch.tensor([[0.5, -1.0, 2.0], [0.5, -1.0, 2.0]])
        assert torch.allclose(y_combined, expected_y)

        # critic1: 0.5∈(0,1) ⇒ True; critic2: -1.0∈(-2,0) & 2.0∈(1,3) ⇒ True ⇒ net_score True
        assert torch.all(net_score == torch.tensor([True, True]))


class TestEquationCritic:
    def test_equation_oracle_forward(self):
        def sum_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        oracle = EquationOracle(sum_func, ["x", "y"], "sum_xy")
        predictions = {"x": torch.tensor([1.0, 2.0]), "y": torch.tensor([3.0, 4.0])}
        result = oracle(predictions)
        assert torch.allclose(result, torch.tensor([[4.0], [6.0]]))
        assert oracle.channels == ["sum_xy"]

    def test_equation_critic_with_limits(self):
        def product_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a * b

        oracle = EquationOracle(product_func, ["x", "y"], "product")
        critic = QuadrantCritic(torch.tensor([0.0]), torch.tensor([10.0]))
        eq_critic = EquationCritic(oracle, critic)

        predictions = {"x": torch.tensor([2.0, 5.0]), "y": torch.tensor([3.0, 3.0])}
        y_out, score = eq_critic(predictions)

        assert torch.allclose(y_out, torch.tensor([[6.0], [15.0]]))
        assert torch.all(score == torch.tensor([True, False]))

    def test_panel_with_equation_critic(self):
        oracle1 = DummyOracle(channels=["a"], output=[2.0])
        limits1 = {"a": (0.0, 5.0)}
        oc1 = OracleCritic(oracle1, QuadrantCritic.from_limits(limits1, ["a"]))

        def double_func(a: torch.Tensor) -> torch.Tensor:
            return 2 * a

        eq_oracle = EquationOracle(double_func, ["a"], "double_a")
        eq_critic_limits = QuadrantCritic(torch.tensor([0.0]), torch.tensor([5.0]))
        eq_critic = EquationCritic(eq_oracle, eq_critic_limits)

        panel = CriticPanel([oc1], [eq_critic])
        assert panel.channels == ["a", "double_a"]

        input_ids = torch.zeros((2, 3), dtype=torch.int64)
        y_combined, net_score = panel(input_ids)

        assert torch.allclose(y_combined, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
        assert torch.all(net_score == torch.tensor([True, True]))
