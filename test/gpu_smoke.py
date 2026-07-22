"""GPU smoke test: a real Dr. GRPO gradient step through DeepSpeed.

The pytest suite in this directory is CPU-only and checks the Dr. GRPO arithmetic in
isolation. This script checks the parts that need a GPU, which is where the fork's
retargeting could still be wrong: DeepSpeed 0.19 against a torch that came from a
vllm>=0.23 resolution, and transformers 5.x loading a causal LM under ZeRO.

It is deliberately not a pytest, so `pytest test/` stays runnable anywhere.

Run it inside a GPU allocation:

    python test/gpu_smoke.py [--model hf-internal-testing/tiny-random-gpt2]

What it exercises, in order:

1. `oat.model.LLM` loads a causal LM, the path `LearnerBase._init` takes.
2. `deepspeed.initialize` wraps it with an optimizer and a ZeRO config.
3. `PPOLearner.get_batch_logps` produces per-token logprobs on device, which is the
   quantity the surrogate's importance ratio is built from.
4. `PPOLearner.compute_monte_carlo_advantages` with `critic_type="drgrpo"` produces the
   mean-centred advantage, and the `grpo` branch differs by the std division.
5. The Dr. GRPO surrogate, `masked_sum` with a constant normalizer, is aggregated,
   backpropagated through DeepSpeed, and stepped.
6. The parameters actually moved and no gradient went non-finite.
"""

import argparse
import os
from types import SimpleNamespace

import deepspeed
import torch

# `deepspeed.initialize` calls `init_distributed`, which falls back to MPI discovery (and
# so imports mpi4py) unless the standard torch.distributed variables are already set. In a
# real run OAT's launcher sets them. This script is a single process, so it declares a
# world of one itself. Must happen before `deepspeed.initialize`.
for _name, _value in (
	("RANK", "0"),
	("LOCAL_RANK", "0"),
	("WORLD_SIZE", "1"),
	("MASTER_ADDR", "127.0.0.1"),
	("MASTER_PORT", "29500"),
):
	os.environ.setdefault(_name, _value)

from oat.algorithms.ppo import PPOLearner
from oat.model import LLM
from oat.utils.ops import masked_sum

GROUP_SIZE = 4
SEQUENCE_LENGTH = 16
MAX_STEPS = 8


def build_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", default="hf-internal-testing/tiny-random-gpt2")
	parser.add_argument("--clip-epsilon", type=float, default=0.2)
	return parser.parse_args()


def main() -> None:
	args = build_args()
	assert torch.cuda.is_available(), "no CUDA device visible; run inside a GPU allocation"
	device = torch.device("cuda")
	print(f"device: {torch.cuda.get_device_name(0)}")
	print(f"torch {torch.__version__} | deepspeed {deepspeed.__version__}")

	# 1. The policy, loaded exactly as LearnerBase._init loads it.
	policy = LLM(args.model, bf16=False, ds_config=None)
	print(f"1. loaded {type(policy.model).__name__}")

	# 2. DeepSpeed wrapping. ZeRO stage 1 keeps this runnable on a single card.
	ds_config = {
		"train_micro_batch_size_per_gpu": GROUP_SIZE,
		"gradient_accumulation_steps": 1,
		"zero_optimization": {"stage": 1},
		"fp16": {"enabled": False},
		"bf16": {"enabled": False},
	}
	# ZeRO needs a real optimizer; passing only `model_parameters` leaves DeepSpeed with a
	# DummyOptim and it asserts. OAT's own strategy builds FusedAdam/DeepSpeedCPUAdam here,
	# but those are JIT-compiled ops, so plain AdamW keeps this smoke test dependency-free.
	engine, optimizer, _, _ = deepspeed.initialize(
		model=policy.model,
		optimizer=torch.optim.AdamW(policy.model.parameters(), lr=1e-4),
		config=ds_config,
	)
	print(f"2. deepspeed.initialize OK, zero stage {ds_config['zero_optimization']['stage']}")

	vocab_size = int(engine.module.config.vocab_size)
	generator = torch.Generator(device="cpu").manual_seed(0)
	input_ids = torch.randint(
		0, vocab_size, (GROUP_SIZE, SEQUENCE_LENGTH), generator=generator
	).to(device)
	attention_mask = torch.ones_like(input_ids)

	# 3. Per-token logprobs, the quantity the importance ratio is built from.
	logprob_stub = SimpleNamespace(args=SimpleNamespace(use_fused_lm_head=False))
	logprobs, _ = PPOLearner.get_batch_logps(
		logprob_stub, engine, input_ids, attention_mask, temperature=1.0
	)
	assert torch.isfinite(logprobs).all(), "non-finite logprobs"
	print(f"3. get_batch_logps OK, shape {tuple(logprobs.shape)}")

	# 4. The Dr. GRPO advantage, and the difference from vanilla GRPO.
	rewards = torch.tensor([[1.0], [0.0], [0.5], [0.25]], device=device)
	advantage_stub = lambda critic: SimpleNamespace(  # noqa: E731
		args=SimpleNamespace(critic_type=critic, num_samples=GROUP_SIZE)
	)
	drgrpo_advantages = PPOLearner.compute_monte_carlo_advantages(
		advantage_stub("drgrpo"), rewards, None
	)
	grpo_advantages = PPOLearner.compute_monte_carlo_advantages(
		advantage_stub("grpo"), rewards, None
	)
	expected = rewards.squeeze(-1) - rewards.mean()
	torch.testing.assert_close(drgrpo_advantages, expected)
	assert not torch.allclose(drgrpo_advantages, grpo_advantages), "critics did not differ"
	print(f"4. drgrpo advantage {drgrpo_advantages.tolist()} (mean-centred, no std division)")

	# 5. The surrogate. Perturb the recorded old logprobs so the ratio is genuinely off 1
	# and the clip actually binds. Leaving old == new would make ratio identically 1 and,
	# because mean-centred advantages sum to zero, drive the loss to exactly 0.0 -- which
	# looks like a pass while exercising neither the ratio nor the clip.
	response_mask = torch.ones_like(logprobs)
	offset = torch.linspace(-0.5, 0.5, logprobs.numel(), device=logprobs.device)
	old_logprobs = (logprobs.detach() + offset.reshape(logprobs.shape))
	ratio = torch.exp(logprobs - old_logprobs)
	assert (ratio < 1 - args.clip_epsilon).any() and (ratio > 1 + args.clip_epsilon).any(), (
		"perturbation did not push the ratio outside the clip range on both sides"
	)
	per_token = torch.minimum(
		ratio * drgrpo_advantages.unsqueeze(-1),
		torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
		* drgrpo_advantages.unsqueeze(-1),
	)
	loss = -masked_sum(per_token, response_mask, axis=1, constant_normalizer=MAX_STEPS).mean()
	assert torch.isfinite(loss), "non-finite loss"
	assert loss.item() != 0.0, "loss is exactly zero; the surrogate was not exercised"
	clipped = ((ratio < 1 - args.clip_epsilon) | (ratio > 1 + args.clip_epsilon)).sum().item()
	print(f"5. dr.grpo surrogate loss {loss.item():.6f} ({clipped}/{ratio.numel()} ratios clipped)")

	# 6. Backward through DeepSpeed and step.
	before = [p.detach().clone() for p in engine.module.parameters() if p.requires_grad]
	engine.backward(loss)
	engine.step()
	after = [p.detach() for p in engine.module.parameters() if p.requires_grad]
	moved = sum(1 for a, b in zip(before, after) if not torch.equal(a, b))
	assert all(torch.isfinite(p).all() for p in after), "non-finite parameter after step"
	assert moved > 0, "optimizer step did not change any parameter"
	print(f"6. backward + step OK, {moved}/{len(before)} parameter tensors moved")

	print("\nGPU SMOKE PASSED")


if __name__ == "__main__":
	main()
