# posit fork of OAT

This fork exists so that [posit](https://github.com/zbambergerNLP/posit) can use OAT's Dr. GRPO on a modern stack. Upstream is [sail-sg/oat](https://github.com/sail-sg/oat), forked at commit `8697066` (2026-01-29).

## Why

posit runs Python 3.12 with `vllm >= 0.23`, `transformers >= 5.6`, and `datasets` (which needs `pyarrow >= 15`). OAT as shipped cannot be installed there.

| Constraint | posit | upstream OAT |
| --- | --- | --- |
| Python | 3.12 | `~=3.10`, and `dm-launchpad` ships no wheel above cp310 |
| vLLM | `>= 0.23.0` | `== 0.11.0` |
| transformers | `>= 5.6` | `== 4.51.3` on PyPI, written against v4 on main |
| pyarrow | `>= 15` via `datasets` | `< 12` |
| numpy, protobuf | resolved via torch | `== 1.26.4`, `== 3.20.0` |

Every one of those pins exists to support the **distributed rollout substrate**, not the algorithm. `dm-launchpad` orchestrates the actor and learner processes. `pyarrow.plasma` carries experience from actor to learner, and was removed from pyarrow in 12.0, which is why the ceiling cannot simply be lifted. `vllm` is pinned because the actor builds an `AsyncLLMEngine` in-process.

posit collects its own rollouts through an OpenEnv environment and reaches models over HTTP against `vllm serve`, so none of that substrate is wanted. Removing it removes the pins.

## What was removed

| Path | Reason |
| --- | --- |
| `oat/actors/` | In-process vLLM generation. The only site importing `AsyncLLMEngine`, `AsyncEngineArgs`, `LoRARequest`. |
| `oat/collectors/` | Pulls experience from actors over Plasma. |
| `oat/utils/ipc.py` | The `pyarrow.plasma` transport. |
| `oat/interface.py` | The Launchpad process topology. |
| `oat/rm/` | Preference reward-model backbones. Held the only `transformers.models.deberta_v2.modeling_deberta_v2` internal import, which does not survive transformers 5. |
| `oat/oracles/remote/` | Mosec-served remote reward model. |
| `oat/experiment/`, `examples/` | Launch scripts, all Launchpad-based. |
| `oat/algorithms/{xpo,apl,ppo_multiturn}.py`, `oat/algorithms/rft.py` | Algorithms coupled to the actor stack. |
| `oat/learners/dap_with_rm.py`, `oat/exploration.py` | Coupled to `oat/rm/`. |
| `PPOActor` in `oat/algorithms/ppo.py` | The generation half of PPO. `PPOLearner` is untouched. |

14,025 lines to 7,935.

## What was kept, and what still works

`PPOLearner` is unchanged, including `compute_monte_carlo_advantages`, `learning_step`, `get_batch_logps`, and the `masked_aggregator` selection. Both Dr. GRPO fixes are byte-for-byte upstream. `LearnerBase`, `RLLearner`, `OfflineLearner`, `SFTLearner`, `oat/model.py`, `oat/utils/*`, and the math and countdown oracles are all retained.

The intended entry point is the **offline path**. `OfflineLearner.run` never touches actors. It iterates a pre-loaded `all_buffer` and calls `learn`, which is exactly the shape a caller with its own rollout loop needs. Upstream reached this via an `__init__` monkeypatch in `run_offline_ppo.py`. Here it is the supported mode. Learner constructors still accept `actors`, which must be empty; passing a non-empty list raises with an explanatory message.

## Changes beyond deletion

- `oat/types.py` gains an `ActorBase` placeholder so the learner signatures that annotate the (now always empty) actor list still typecheck.
- `oat/model.py` passes `dtype=` rather than `torch_dtype=` to `from_pretrained`. transformers 5 deprecated the old spelling.
- `pyproject.toml` targets `>= 3.12`, drops `dm-launchpad`, `mosec`, `vllm`, `pyarrow`, and `flash-attn`, and converts the remaining exact pins to floors. Distribution name is `oat-llm-posit` so it cannot be confused with upstream in `pip list`. The import name stays `oat`.

## Known wart carried over

`oat/utils/math_grader.py` has invalid escape sequences in regex literals, which Python 3.12 reports as `SyntaxWarning` and Python 3.14 will reject. They are left alone deliberately. Several are entangled with valid `\\` escapes, so a mechanical fix silently changes what the patterns match, and posit does not use the math grader.

## Tests

`test/test_drgrpo_semantics.py` pins both Dr. GRPO fixes numerically. Neither is a distinct class upstream, both are conditionals on `args.critic_type`, so they are easy to perturb during a rebase. The tests assert that

- the `drgrpo` advantage is exactly the reward minus its group mean,
- `grpo` differs from it by exactly the group standard-deviation division,
- a group whose rollouts all score alike yields exactly zero advantage under `drgrpo` while `grpo` divides by `0 + 1e-8`,
- the std division equalizes the pull of a wide-spread and a narrow-spread prompt, which is the difficulty bias,
- `masked_sum` with a constant normalizer weights every token equally while `masked_mean` weights every sequence equally, which is the length bias,
- the choice of constant is a global rescale that folds into the learning rate.

Run the hermetic parity suite with `pytest test/test_drgrpo_semantics.py`. Run
`test/test_trajectory_dataset.py` separately when its Hugging Face model and dataset are
available in a writable cache.

## Verified

Current POSIT environment: Python 3.12.13, torch 2.11.0, transformers 5.14.1,
numpy 2.3.5, and deepspeed 0.19.2.

- All 33 retained modules import.
- The 11 hermetic tests in `test/test_drgrpo_semantics.py` pass. The inherited
  `test_trajectory_dataset.py` is a Hugging Face dataset integration test and requires
  network plus a writable cache (or a prepared offline cache); it is not a hermetic fork test.
- Commit `c4858c9ed87cd0794a7218c37f8fdd3bc085ab24` installs under the distribution name
  `oat-llm-posit` into a fresh Python 3.12 virtual environment.
- Ionic A6000 job `30048441` loaded `oat.model.LLM`, initialized DeepSpeed ZeRO-1,
  produced finite `PPOLearner.get_batch_logps`, computed the mean-centered Dr. GRPO
  advantage and clipped surrogate, backpropagated, and changed 64/64 parameter tensors.

The GPU check is intentionally a smoke test. It assembles the surrogate directly and does
not invoke `OfflinePPOLearner.run`, `prepare_data`, or the complete
`PPOLearner.learning_step`. POSIT will run that learner-level end-to-end validation only
after its Phase 4 trajectory-to-`TransitionData` adapter exists.
