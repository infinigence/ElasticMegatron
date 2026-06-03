import torch

from ..distributed.elastic_process_group import (
    ElasticProcessGroup,
    create_p2p_collective_groups,
    get_p2p_collective_group,
)


class P2PToCollective:
    """
    Convert P2P (point-to-point) communication to collective communication using
    2-rank broadcast groups.

    Important: This implementation uses `torch.distributed.broadcast` which is a
    collective operation. Therefore, both sender and receiver MUST call their
    respective send/recv methods simultaneously (or use async_op=True and wait).
    Mismatched calls will cause the program to hang.

    Example:
        # Rank 0 sends to rank 1
        # Rank 0 side:
        p2p.send(tensor, dst_rank=1)

        # Rank 1 side (must be called simultaneously):
        p2p.recv(tensor, src_rank=0)
    """

    def __init__(self):
        self._rank = torch.distributed.get_rank()
        create_p2p_collective_groups()

    def _get_p2p_collective_group(self, rank: int) -> ElasticProcessGroup:
        """
        Get the 2-rank NCCL group for communication with the given rank.

        Note: ranks are normalized to ensure consistent group lookup regardless
        of the order (e.g., [rank0, rank1] vs [rank1, rank0]).
        """
        # Normalize ranks to ensure consistent group lookup
        # (ElasticProcessGroupManager uses frozenset internally, but this makes it explicit)
        ranks = [min(self._rank, rank), max(self._rank, rank)]
        return get_p2p_collective_group(ranks=ranks, backend="nccl")

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
        async_op: bool = False,
    ):
        """
        Send a tensor to the destination rank using broadcast collective.

        WARNING: The destination rank MUST call recv() simultaneously, otherwise
        this will hang because broadcast requires all ranks in the group to participate.

        Args:
            tensor: Tensor to send. Will be modified in-place by broadcast.
            dst_rank: Destination rank to send to.
            async_op: If True, returns an async work handle instead of blocking.

        Returns:
            AsyncWork object if async_op=True, otherwise None.
        """
        if dst == self._rank:
            raise ValueError("Cannot send to self")

        group = self._get_p2p_collective_group(dst)
        # Broadcast with src=self._rank: data flows from self._rank to dst_rank

        return torch.distributed.broadcast(
            tensor=tensor, group=group.group, src=self._rank, async_op=async_op
        )

    def recv(self, tensor: torch.Tensor, src: int, async_op: bool = False):
        """
        Receive a tensor from the source rank using broadcast collective.

        WARNING: The source rank MUST call send() simultaneously, otherwise
        this will hang because broadcast requires all ranks in the group to participate.

        Args:
            tensor: Tensor to receive into. Will be modified in-place by broadcast.
            src_rank: Source rank to receive from.
            async_op: If True, returns an async work handle instead of blocking.

        Returns:
            AsyncWork object if async_op=True, otherwise None.
        """
        if src == self._rank:
            raise ValueError("Cannot receive from self")

        group = self._get_p2p_collective_group(src)
        # Broadcast with src=src_rank: data flows from src_rank to self._rank
        return torch.distributed.broadcast(
            tensor=tensor, group=group.group, src=src, async_op=async_op
        )
