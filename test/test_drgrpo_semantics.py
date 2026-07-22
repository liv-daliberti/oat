# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pin the two Dr. GRPO fixes numerically.

Dr. GRPO (Liu et al. 2025, arXiv:2503.20783) is GRPO with two optimization biases removed.
In this codebase neither is a distinct loss class; both are conditionals on
``args.critic_type``, which makes them easy to perturb by accident during a rebase. These
tests assert the exact arithmetic of each.

Fix 1, response-level length bias. ``PPOLearner._init`` picks the loss aggregator:
``masked_sum`` with a constant normalizer for ``drgrpo``, ``masked_mean`` otherwise.

Fix 2, question-level difficulty bias. ``PPOLearner.compute_monte_carlo_advantages``
divides the centred group reward by the group standard deviation for ``grpo`` and skips
that division for ``drgrpo``.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from oat.algorithms.ppo import PPOLearner
from oat.utils.ops import masked_mean, masked_sum

# The learner method under test reads only these two fields off ``self``.
def _learner(critic_type: str, num_samples: int) -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(critic_type=critic_type, num_samples=num_samples)
    )


def _advantages(critic_type: str, group_rewards: list[list[float]]) -> torch.Tensor:
    """Run ``compute_monte_carlo_advantages`` over groups laid out contiguously.

    The method assumes the ``num_samples`` rollouts of each prompt sit adjacent in the
    buffer, which is why the dataloader disables shuffling for the group MC critics.
    """
    num_samples = len(group_rewards[0])
    assert all(len(g) == num_samples for g in group_rewards)
    flat = [r for group in group_rewards for r in group]
    # Rewards arrive token-shaped and are summed over the sequence axis internally.
    rewards = torch.tensor(flat, dtype=torch.float32).unsqueeze(-1)
    return PPOLearner.compute_monte_carlo_advantages(
        _learner(critic_type, num_samples), rewards, response_masks=None
    )


# ---------------------------------------------------------------------------- #
# Fix 2: the advantage drops the group standard deviation.
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "group_rewards",
    [
        pytest.param([[1.0, 0.0, 0.0, 0.0]], id="one-winner"),
        pytest.param([[1.0, 1.0, 1.0, 0.0]], id="one-loser"),
        pytest.param([[0.9, 0.5, 0.2, 0.1]], id="graded"),
        pytest.param([[1.0, 0.0], [0.7, 0.3], [0.5, 0.5]], id="several-groups"),
    ],
)
def test_drgrpo_advantage_is_group_mean_centred(group_rewards):
    """Dr. GRPO's advantage is exactly the reward minus its group mean."""
    got = _advantages("drgrpo", group_rewards)
    expected = torch.tensor(
        [r - sum(group) / len(group) for group in group_rewards for r in group]
    )
    torch.testing.assert_close(got, expected)


def test_drgrpo_advantages_sum_to_zero_within_each_group():
    """Mean-centring makes each group's advantages sum to zero, so the baseline is exact."""
    got = _advantages("drgrpo", [[1.0, 0.0, 0.4, 0.2], [0.3, 0.3, 0.9, 0.1]])
    torch.testing.assert_close(
        got.view(2, 4).sum(dim=1), torch.zeros(2), atol=1e-6, rtol=0
    )


def test_grpo_divides_by_group_std_and_drgrpo_does_not():
    """The only difference between the two critics is the standard-deviation division."""
    group = [[1.0, 0.0, 0.5, 0.25]]
    dr = _advantages("drgrpo", group)
    vanilla = _advantages("grpo", group)
    # torch.std is unbiased (n-1), matching what the learner computes.
    std = torch.tensor(group[0]).std()
    torch.testing.assert_close(vanilla, dr / (std + 1e-8))
    assert not torch.allclose(vanilla, dr)


def test_degenerate_group_is_zero_under_drgrpo_but_explodes_under_grpo():
    """A group whose rollouts all score alike is the case the std division cannot handle.

    Dr. GRPO returns exactly zero advantage, which correctly contributes no gradient. GRPO
    divides zero by ``0 + 1e-8``, which is numerically defined but leaves the update at the
    mercy of float error on a group that carries no signal.
    """
    dr = _advantages("drgrpo", [[0.6, 0.6, 0.6, 0.6]])
    torch.testing.assert_close(dr, torch.zeros(4))

    # Near-degenerate: a spread of 1e-6 is amplified by six orders of magnitude.
    tiny = [[0.5, 0.5, 0.5, 0.500001]]
    dr_tiny = _advantages("drgrpo", tiny)
    grpo_tiny = _advantages("grpo", tiny)
    assert dr_tiny.abs().max() < 1e-5
    assert grpo_tiny.abs().max() > 1.0


def test_std_division_upweights_low_variance_prompts():
    """The difficulty bias, stated directly.

    Two prompts, same ordering of outcomes, different spread. Dr. GRPO keeps the wide-spread
    prompt more influential. GRPO rescales both to comparable magnitude, so the prompt that
    discriminates least between its rollouts pulls just as hard on the update.
    """
    wide, narrow = [1.0, 0.0], [0.55, 0.45]
    dr = _advantages("drgrpo", [wide, narrow])
    vanilla = _advantages("grpo", [wide, narrow])

    dr_wide, dr_narrow = dr[:2].abs().max(), dr[2:].abs().max()
    gr_wide, gr_narrow = vanilla[:2].abs().max(), vanilla[2:].abs().max()

    assert dr_wide > 5 * dr_narrow
    torch.testing.assert_close(gr_wide, gr_narrow)


# ---------------------------------------------------------------------------- #
# Fix 1: the surrogate normalizes by a constant, not by each sequence's length.
# ---------------------------------------------------------------------------- #


def _aggregators(max_length: int):
    """The two branches of ``PPOLearner._init``'s ``masked_aggregator`` choice."""
    drgrpo = functools.partial(masked_sum, constant_normalizer=max_length)
    return drgrpo, masked_mean


def test_constant_normalizer_weights_every_token_equally():
    """Two responses of different length, identical per-token loss.

    ``masked_mean`` returns the same value for both, so each response contributes equally
    however many tokens it spent. ``masked_sum`` over a constant returns values in
    proportion to length, so each *token* contributes equally. The latter is the unbiased
    per-token estimator.
    """
    max_length = 8
    values = torch.ones(2, max_length)
    mask = torch.zeros(2, max_length)
    mask[0, :2] = 1.0  # a short response
    mask[1, :6] = 1.0  # a long one

    drgrpo, vanilla = _aggregators(max_length)
    torch.testing.assert_close(vanilla(values, mask, axis=1), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(
        drgrpo(values, mask, axis=1), torch.tensor([2 / 8, 6 / 8])
    )


def test_mean_normalizer_inflates_per_token_gradient_on_short_sequences():
    """The length bias, stated directly.

    Under ``masked_mean`` a token inside a two-token response carries three times the weight
    of a token inside a six-token response. That asymmetry is what pushes wrong answers
    longer, since a negative advantage is diluted by generating more tokens.
    """
    max_length = 8
    values = torch.ones(2, max_length)
    mask = torch.zeros(2, max_length)
    mask[0, :2] = 1.0
    mask[1, :6] = 1.0

    per_token_weight_under_mean = 1.0 / mask.sum(dim=1)
    assert per_token_weight_under_mean[0] == pytest.approx(3 * per_token_weight_under_mean[1])

    drgrpo, _ = _aggregators(max_length)
    # Under the constant normalizer every token is worth 1/max_length regardless.
    torch.testing.assert_close(
        drgrpo(values, mask, axis=1) / mask.sum(dim=1),
        torch.full((2,), 1 / max_length),
    )


def test_constant_normalizer_is_a_global_rescale_of_the_plain_sum():
    """Choosing the constant only rescales the objective, so it folds into the learning rate."""
    values = torch.randn(4, 8)
    mask = (torch.arange(8).unsqueeze(0) < torch.tensor([[2], [4], [6], [8]])).float()
    plain = masked_sum(values, mask, axis=1, constant_normalizer=1.0)
    for constant in (2.0, 8.0, 512.0):
        torch.testing.assert_close(
            masked_sum(values, mask, axis=1, constant_normalizer=constant),
            plain / constant,
        )
