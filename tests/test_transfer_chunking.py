"""Behavioral test for the rank-invariant reshard-transfer chunker.

Chunk boundaries MUST depend only on the global vparam list (vp.size), never on
per-rank ownership, or the per-peer NCCL butterfly desyncs. Importing the module
needs torch + megatron, so this SKIPS where they are unavailable (runs on the A100
box). Unlike a string-grep guard, it constructs the inputs and asserts behaviour.
"""


def _load():
    try:
        from elastic_megatron.transfer.transfer import TransferManager

        return TransferManager
    except Exception:  # torch / megatron / cpu_offloading not importable here
        return None


class _VP:
    def __init__(self, size):
        self.size = size


def test_chunking_deterministic_budget_bounded_and_degenerate():
    TransferManager = _load()
    if TransferManager is None:
        print("SKIP test_chunking (torch/megatron unavailable)")
        return
    tm = TransferManager.__new__(TransferManager)  # bypass __init__ (no dist needed)
    vps = [_VP(100), _VP(100), _VP(100), _VP(50)]

    chunks = list(tm._chunk_virtual_params(vps, budget_numel=200))
    assert [[v.size for v in c] for c in chunks] == [[100, 100], [100, 50]]
    # deterministic: identical inputs -> identical boundaries (rank-invariance proxy)
    assert [[v.size for v in c] for c in tm._chunk_virtual_params(vps, 200)] == [
        [100, 100],
        [100, 50],
    ]
    # an oversized single vparam still goes alone (never split, never dropped)
    big = [_VP(10_000), _VP(10)]
    assert [[v.size for v in c] for c in tm._chunk_virtual_params(big, 200)] == [
        [10_000],
        [10],
    ]
    # degenerate: None / <=0 budget -> exactly one chunk (today's batched behaviour)
    assert [[v.size for v in c] for c in tm._chunk_virtual_params(vps, None)] == [
        [100, 100, 100, 50]
    ]
    assert [[v.size for v in c] for c in tm._chunk_virtual_params(vps, 0)] == [
        [100, 100, 100, 50]
    ]


if __name__ == "__main__":
    test_chunking_deterministic_budget_bounded_and_degenerate()
    print("PASS tests/test_transfer_chunking.py")
