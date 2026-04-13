from typing import List, Tuple
import torch
import time

from packaging import version

VERSION_MINOR = None


def get_megatron_version_minor() -> int:
    global VERSION_MINOR
    if VERSION_MINOR is None:
        from megatron.core import __version__ as megatron_version

        VERSION_MINOR = version.parse(megatron_version).minor
    return VERSION_MINOR


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, unit: str = "ms"):
        """
        Args:
            unit: Time unit ('ms' for milliseconds, 's' for seconds, 'ns' for nanoseconds)
        """
        self.unit = unit
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        end_time = time.perf_counter_ns()
        elapsed_ns = end_time - self.start_time
        if self.unit == "ms":
            self.elapsed = elapsed_ns / 1e6
        elif self.unit == "s":
            self.elapsed = elapsed_ns / 1e9
        else:  # 'ns'
            self.elapsed = elapsed_ns
        return False


class Range:
    """
    Copy from megatron.core.optimizer.distrib_optimizer.
    Add method : __eq__() and chunk()

    A range represents a start and end points for indexing a shard
    from a full tensor.
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start
        self.is_normalized = start == 0

    def normalize(self, global_offset: int = None) -> "Range":
        if global_offset is None:
            global_offset = self.start
        assert global_offset <= self.start
        return Range(self.start - global_offset, self.end - global_offset)

    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)

    def __len__(self):
        return self.end - self.start

    def __eq__(self, other):
        assert isinstance(other, Range)
        return other.start == self.start and other.end == self.end

    def chunk(self, split: int):
        assert split >= 1 and self.size % split == 0
        split_size = self.size // split
        return [
            Range(self.start + i * split_size, self.start + (i + 1) * split_size)
            for i in range(split)
        ]

    def is_contain(self, other):
        assert isinstance(other, Range)
        return other.start >= self.start and other.end <= self.end

    def __hash__(self):
        return hash((self.start, self.end))

    def get_sub_range_slices(self, sub_range: "Range") -> tuple[slice]:
        assert self.is_normalized
        assert isinstance(sub_range, Range)

        if self == sub_range:
            return tuple([slice(None)])

        assert self.is_contain(sub_range)
        return tuple([slice(sub_range.start, sub_range.end)])

    def __str__(self):
        return f"Range([{self.start},{self.end}) [size={self.size}])"

    def __repr__(self):
        return str(self)


class ParamRange:
    def __init__(
        self,
        ranges: List[Range] = None,
        param_shape: torch.Size = None,
    ):
        self.shapes: List[Range] = []
        if param_shape is not None:
            for _size in param_shape:
                self.shapes.append(Range(0, _size))
        else:
            self.shapes = ranges
        self.dims = len(self.shapes)
        assert self.dims <= 2

        self.is_normalized = True
        for shape in self.shapes:
            if not shape.is_normalized:
                self.is_normalized = False
                break

        self.size = 1
        for shape in self.shapes:
            self.size *= shape.size

    def normalize(self, global_offsets: List[int] = None) -> "ParamRange":
        if global_offsets is None:
            global_offsets = [0] * self.dims
        else:
            assert len(global_offsets) == self.dims
        new_shapes = []
        for i, shape in enumerate(self.shapes):
            new_shapes.append(shape.normalize(global_offsets[i]))
        return ParamRange(ranges=new_shapes)

    def to_torch_size(self) -> torch.Size:
        return torch.Size([shape.size for shape in self.shapes])

    def chunk(self, split: int, dim: int):
        assert 0 <= dim < self.dims
        dim_range_views: List[Range] = self.shapes[dim].chunk(split=split)
        views = []
        for view in dim_range_views:
            new_shapes = list(self.shapes)
            new_shapes[dim] = view
            views.append(ParamRange(ranges=new_shapes))
        return views

    def is_contain(self, other: "ParamRange") -> int:
        """
        Requires the number of dims to be the same, and only one shape can be a sub-range, the remaining shapes must be equal.
        Returns the dim which corresponds to the sub-range.
        """
        assert isinstance(other, ParamRange)
        if self.dims != other.dims:
            return -1
        sub_dim = -1
        for dim, self_shape in enumerate(self.shapes):
            other_shape = other.shapes[dim]
            if self_shape == other_shape:
                continue
            # Note: only one sub-dimension can be different
            if sub_dim != -1:
                return -1
            if self_shape.is_contain(other_shape):
                sub_dim = dim
        return sub_dim

    def get_sub_range_slices(self, sub_range: "ParamRange") -> Tuple[slice]:
        """Get sub_range slices."""
        assert self.is_normalized
        assert isinstance(sub_range, ParamRange)

        slices = [slice(None)] * self.dims
        if self == sub_range:
            return tuple(slices)

        sub_dim = self.is_contain(sub_range)
        assert sub_dim != -1
        slices[sub_dim] = slice(
            sub_range.shapes[sub_dim].start,
            sub_range.shapes[sub_dim].end,
        )
        return tuple(slices)

    def __eq__(self, other):
        assert isinstance(other, ParamRange)
        if self.dims != other.dims:
            return False

        for self_shape, other_shape in zip(self.shapes, other.shapes):
            if self_shape != other_shape:
                return False
        return True

    def __len__(self):
        return self.size

    def __str__(self):
        msg = "ParamRange("
        for i, shape in enumerate(self.shapes):
            msg += f"dim{i} = {str(shape)}  "
        msg += ")"
        return msg

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.dims, tuple(self.shapes)))
