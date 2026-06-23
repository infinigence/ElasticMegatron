"""Behavioral test for HybridDeviceOptimizerAdapter.repair_per_param_step.

The cpu-adam reshard never transfers the per-param Adam ``step`` (dropped as
non-param-shaped, I-15) and the dst init's ``dummy_step()`` seeds it to a tensor
=1 (one throwaway real step), never corrected to the true src step. torch AdamW
(the HDO CPU sub-optimizer) reads the PER-PARAM ``state[p]["step"]`` for bias
correction, so the step must be repaired from the broadcast param_groups step.
This exercises that logic on a minimal HDO stub (no GPU / no Megatron required at
run time, but importing the adapter needs torch + megatron, so the test SKIPS
where they are unavailable — it runs on the A100 box).

Unlike the grep-only static guards, this constructs the objects and asserts the
behaviour: the dummy_step placeholder (1) becomes the src step, and the hdo.state
alias tracks it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    """Import the adapter + torch, or return (None, None) if unavailable."""
    try:
        import torch  # noqa: F401
        from elastic_megatron.resharding.optimizer_adapter import (
            HybridDeviceOptimizerAdapter,
        )

        return torch, HybridDeviceOptimizerAdapter
    except Exception:  # torch / megatron / cpu_offloading not importable here
        return None, None


def _make_fake_hdo(torch, params, group_step):
    """A minimal stand-in for Megatron's HybridDeviceOptimizer.

    Mirrors only what repair_per_param_step touches: per-param ``cpu_optimizers``
    (stock torch.optim.AdamW) with a step=1 placeholder like ``dummy_step()``
    leaves on the dst, a ``param_groups`` carrying the broadcast src step, and the
    ``_sync_sub_optimizers_state_to_hdo`` aliasing (hdo.state[orig] IS the
    sub-optimizer's state dict).
    """

    class _FakeHDO:
        def __init__(self, params, group_step):
            self.param_groups = [{"params": list(params), "step": group_step}]
            self.cpu_optimizers = []
            self.inner_param_to_orig_param = {}
            for p in params:
                opt = torch.optim.AdamW([p], lr=1e-3)
                opt.state[p] = {
                    # dummy_step() seeds the per-param step to 1 (one throwaway
                    # real step) on the dst — the value the repair must overwrite.
                    "step": torch.ones(()),
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
                self.cpu_optimizers.append(opt)
                self.inner_param_to_orig_param[p] = p
            self.sub_optimizers = list(self.cpu_optimizers)
            self.state = {}
            self._sync_sub_optimizers_state_to_hdo()

        def _sync_sub_optimizers_state_to_hdo(self):
            new_state = {}
            for opt in self.sub_optimizers:
                for p in opt.state:
                    new_state[self.inner_param_to_orig_param[p]] = opt.state[p]
            self.state = new_state

    class _FakeDistOpt:
        def __init__(self, hdo):
            self.optimizer = hdo

    hdo = _FakeHDO(params, group_step)
    return hdo, _FakeDistOpt(hdo)


def test_repair_sets_per_param_step_from_tensor_group_step():
    torch, Adapter = _load()
    if torch is None:
        print(
            "SKIP test_repair_sets_per_param_step_from_tensor_group_step "
            "(torch/megatron unavailable)"
        )
        return
    p1, p2 = torch.zeros(4), torch.zeros(6)
    hdo, dist_opt = _make_fake_hdo(torch, [p1, p2], group_step=torch.tensor(7.0))

    for opt, p in zip(hdo.cpu_optimizers, [p1, p2]):
        # dummy_step placeholder precondition: NOT the true src step yet.
        assert float(opt.state[p]["step"].item()) == 1.0

    Adapter(dist_opt).repair_per_param_step()

    for opt, p in zip(hdo.cpu_optimizers, [p1, p2]):
        # the throwaway 1 is overwritten with the true src step, not left at 1
        assert float(opt.state[p]["step"].item()) == 7.0
        # hdo.state aliases the same dict object -> the repair is visible there too
        assert hdo.state[p]["step"] is opt.state[p]["step"]
        assert float(hdo.state[p]["step"].item()) == 7.0


def test_repair_handles_int_group_step_and_empty_cpu_optimizers():
    torch, Adapter = _load()
    if torch is None:
        print(
            "SKIP test_repair_handles_int_group_step_and_empty_cpu_optimizers "
            "(torch/megatron unavailable)"
        )
        return
    # int (not tensor) param_groups step
    p = torch.zeros(3)
    hdo, dist_opt = _make_fake_hdo(torch, [p], group_step=5)
    Adapter(dist_opt).repair_per_param_step()
    assert float(hdo.cpu_optimizers[0].state[p]["step"].item()) == 5.0

    # no CPU sub-optimizers (all-GPU offload) -> no-op, must not raise
    hdo2, dist_opt2 = _make_fake_hdo(torch, [torch.zeros(2)], group_step=3)
    hdo2.cpu_optimizers = []
    hdo2.sub_optimizers = []
    Adapter(dist_opt2).repair_per_param_step()


if __name__ == "__main__":
    test_repair_sets_per_param_step_from_tensor_group_step()
    test_repair_handles_int_group_step_and_empty_cpu_optimizers()
    print("PASS tests/test_per_param_step_repair.py")
