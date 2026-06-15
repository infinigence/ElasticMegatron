"""Probe: 打包/解包按字节位级往返（torch CPU, <30s, 无需 GPU/分布式 init）。

构造混合 dtype/设备语义的切片列表，按多种 cap 切批：
pack_chunk 进 uint8 staging → unpack_chunk 到零化副本 → 位级一致。
覆盖 chunk 边界两端不一致 / 切片跨 chunk 损坏的风险。

需要 torch；在无 torch 的开发机上跳过，由 A100 箱执行。
"""
import sys

import torch

sys.path.insert(0, ".")
from elastic_megatron.transfer.chunk_schedule import chunk_ranges
from elastic_megatron.transfer.communicator import BatchedTransfer


def _roundtrip(slices: list[torch.Tensor], cap: int | None) -> None:
    src_bytes = [t.contiguous().view(torch.uint8).reshape(-1) for t in slices]
    sizes = [b.numel() for b in src_bytes]
    # 接收落点按生产契约是连续的(communicator: "recv landings are contiguous
    # by contract")——zeros_like 会保留转置 stride,这里显式取连续布局。
    dst = [torch.zeros_like(t.contiguous()) for t in slices]
    dst_bytes = [t.view(torch.uint8).reshape(-1) for t in dst]
    total = sum(sizes)
    for chunk in chunk_ranges(total, cap):
        n = chunk[1] - chunk[0]
        stage = torch.empty(n, dtype=torch.uint8)  # CPU staging 等价物
        BatchedTransfer._pack_chunk(src_bytes, sizes, chunk, stage)
        BatchedTransfer._unpack_chunk(dst_bytes, sizes, chunk, stage)
    for a, b in zip(slices, dst):
        assert torch.equal(a.contiguous().view(torch.uint8), b.view(torch.uint8)), (
            f"roundtrip mismatch: shape={a.shape} dtype={a.dtype} cap={cap}"
        )


def main() -> None:
    g = torch.Generator().manual_seed(0)
    slices = [
        torch.randn(37, 5, generator=g, dtype=torch.float32),
        torch.randn(1, generator=g, dtype=torch.float32).to(torch.bfloat16),
        torch.empty(0, dtype=torch.float32),                       # 0 字节切片
        torch.randn(257, generator=g, dtype=torch.float32),
        torch.randn(8, 8, generator=g, dtype=torch.float32).t(),   # 非连续发送视图
    ]
    for cap in [None, 0, 1, 13, 100, 1 << 20]:
        _roundtrip(slices, cap)
    print("PASS tests/test_packed_chunk_roundtrip.py")


if __name__ == "__main__":
    main()
