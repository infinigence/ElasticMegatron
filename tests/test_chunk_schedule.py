"""Probe: chunk_schedule 纯逻辑（torch-free, <5s）。

验证 BatchedTransfer 打包路径的切批数学：
1) chunk_ranges 精确覆盖 [0, total) 且每段 <= cap；
2) slice_spans 把字节区间映射回切片内 span，覆盖无重叠无遗漏；
3) 调度对称性：同一 (切片字节列表, cap) 在收发两端推出完全相同的批边界
   （两端各自本地推出相同边界，故打包路径无需任何集合通信）。
结构化判据：assert + exit code（probe 契约）。
"""
import importlib.util
import os
import random

# Load the module directly by path: the parent ``elastic_megatron`` package
# __init__ imports torch, which would defeat this probe's torch-free contract
# (it must run on any machine). Loading the leaf file in isolation keeps the
# module under test pure-python while exercising the real source file.
_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "elastic_megatron",
    "transfer",
    "chunk_schedule.py",
)
_spec = importlib.util.spec_from_file_location("chunk_schedule", _MODULE_PATH)
_chunk_schedule = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunk_schedule)
chunk_ranges = _chunk_schedule.chunk_ranges
derive_staging_cap = _chunk_schedule.derive_staging_cap
slice_spans = _chunk_schedule.slice_spans


def test_chunk_ranges():
    assert chunk_ranges(0, 4) == []
    assert chunk_ranges(10, None) == [(0, 10)]
    assert chunk_ranges(10, 0) == [(0, 10)]          # cap<=0 => 单 chunk（回退语义）
    assert chunk_ranges(10, -1) == [(0, 10)]
    assert chunk_ranges(10, 100) == [(0, 10)]
    assert chunk_ranges(10, 5) == [(0, 5), (5, 10)]   # 整除
    assert chunk_ranges(11, 4) == [(0, 4), (4, 8), (8, 11)]  # 余数
    # 不变量：精确覆盖、不重叠、每段<=cap
    for total, cap in [(1, 1), (1000, 7), (4096, 4096), (4097, 4096)]:
        rs = chunk_ranges(total, cap)
        assert rs[0][0] == 0 and rs[-1][1] == total
        assert all(e - s <= cap for s, e in rs)
        assert all(rs[i][1] == rs[i + 1][0] for i in range(len(rs) - 1))


def test_slice_spans():
    sizes = [4, 0, 6, 2]  # 含 0 字节切片
    # 区间落在单切片内
    assert slice_spans(sizes, 1, 3) == [(0, 1, 2)]
    # 跨多切片（跳过 0 字节切片）
    assert slice_spans(sizes, 2, 11) == [(0, 2, 2), (2, 0, 6), (3, 0, 1)]
    # 精确边界
    assert slice_spans(sizes, 0, 12) == [(0, 0, 4), (2, 0, 6), (3, 0, 2)]
    assert slice_spans(sizes, 4, 10) == [(2, 0, 6)]
    # span 字节总和 == 区间长度
    for start, end in [(0, 12), (3, 9), (5, 6)]:
        assert sum(n for _, _, n in slice_spans(sizes, start, end)) == end - start


def test_schedule_symmetry():
    """发送端的 send 列表与接收端的 recv 落地列表 = 同序同尺寸切片，
    同 cap 推出的 (chunk 边界, 每 chunk 的 span 集) 必须逐字节一致。"""
    rng = random.Random(0)
    for _ in range(200):
        sizes = [rng.randrange(0, 50) for _ in range(rng.randrange(1, 12))]
        cap = rng.choice([None, 0, 1, 7, 64, sum(sizes) or 1])
        total = sum(sizes)
        sender = [(c, slice_spans(sizes, *c)) for c in chunk_ranges(total, cap)]
        receiver = [(c, slice_spans(sizes, *c)) for c in chunk_ranges(total, cap)]
        assert sender == receiver
        # 全部 span 拼起来恰好覆盖每个切片的每个字节
        covered = {i: 0 for i in range(len(sizes))}
        for _, spans in sender:
            for idx, off, n in spans:
                assert covered[idx] == off  # 顺序、无缝
                covered[idx] += n
        assert all(covered[i] == sizes[i] for i in range(len(sizes)))


def test_derive_staging_cap():
    GiB = 1 << 30
    MiB = 1 << 20
    FLOOR = 512 * MiB  # hard staging-cap floor
    RESERVE = 2 * GiB  # default headroom subtracted before halving
    # 无可用量信息 / hygiene 关 → 下限 (512 MiB)
    assert derive_staging_cap(None) == FLOOR
    assert derive_staging_cap(0) == FLOOR
    # available <= reserve → usable<=0 → 钳到下限 (512 MiB)
    assert derive_staging_cap(MiB) == FLOOR
    assert derive_staging_cap(2 * GiB) == FLOOR  # (2-2)//2=0 → floor
    # 正常区间：cap = (available − 2GiB 余量) / 2（发+收两块货位共存 + 留余量）
    assert derive_staging_cap(3 * GiB) == FLOOR          # (3-2)//2 = 512MiB = floor
    assert derive_staging_cap(5 * GiB) == 3 * GiB // 2   # 用户场景:5GB→1.5GiB,2×cap=3GiB,留2GiB
    assert derive_staging_cap(10 * GiB) == 4 * GiB       # (10-2)//2 = 4GiB
    # 余量保证 2×cap 不打满 available
    assert 2 * derive_staging_cap(5 * GiB) == 5 * GiB - RESERVE
    # 超大模型 → 钳到 8 GiB 上限
    assert derive_staging_cap(100 * GiB) == 8 * GiB      # (100-2)//2=49 → ceil
    # 自定义钳位透传（reserve=0 隔离 clamp 逻辑）
    assert derive_staging_cap(10, floor=10, ceil=20, reserve=0) == 10   # 10//2=5 → floor
    assert derive_staging_cap(30, floor=10, ceil=20, reserve=0) == 15   # 区间内
    assert derive_staging_cap(100, floor=10, ceil=20, reserve=0) == 20  # 100//2=50 → ceil


if __name__ == "__main__":
    test_chunk_ranges()
    test_slice_spans()
    test_schedule_symmetry()
    test_derive_staging_cap()
    print("PASS tests/test_chunk_schedule.py")
