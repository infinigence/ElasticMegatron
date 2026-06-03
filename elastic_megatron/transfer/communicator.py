import queue
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist


@dataclass
class CommunicationBytes:
    send: Dict[int, int]
    recv: Dict[int, int]
    total: int


class BatchP2P:
    """Accumulate p2p sends/recvs and dispatch them in a single ``batch_isend_irecv``.

    This is the batched counterpart to :meth:`Communicator.send` /
    :meth:`Communicator.recv`. It shares the same behaviors so call sites never
    need to construct :class:`torch.distributed.P2POp` themselves:

    * ``fake_transfer`` mode: pre-builds the NCCL connection instead of sending.
    * Self-rank send/recv: routed through ``Communicator``'s self-copy queue.
    * Non-contiguous tensors: send via a contiguous shadow buffer; recv into a
      contiguous temp buffer that is copied back on :meth:`wait`.
    * Communication bytes accounting matches :meth:`Communicator.send` /
      :meth:`Communicator.recv`.

    The order of :meth:`isend` / :meth:`irecv` calls is preserved in the
    resulting op list, so callers can still control op ordering to avoid
    deadlocks (e.g. send-before-recv on the lower-ranked peer).
    """

    def __init__(self, communicator: "Communicator"):
        self._communicator = communicator
        self._p2p_ops: List[dist.P2POp] = []
        # Keep references to send-side shadow buffers until wait() so the
        # underlying storage is not freed while the isend is in flight.
        self._send_buffers: List[torch.Tensor] = []
        self._recv_copy_back: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def isend(self, tensor: torch.Tensor, dst: int) -> None:
        comm = self._communicator
        if comm._build_nccl_connection_only:
            comm._build_p2p_connection(dst, comm.send_fn)
            return
        if dst == comm._rank:
            comm._send_self(tensor.clone().detach().contiguous())
            return

        send_buffer = tensor.contiguous()
        self._send_buffers.append(send_buffer)
        self._p2p_ops.append(dist.P2POp(dist.isend, send_buffer, dst))
        comm._record_send_bytes(dst, send_buffer.nbytes)

    def irecv(self, tensor: torch.Tensor, src: int) -> None:
        comm = self._communicator
        if comm._build_nccl_connection_only:
            comm._build_p2p_connection(src, comm.recv_fn)
            return
        if src == comm._rank:
            comm._recv_self(tensor)
            return

        if tensor.is_contiguous():
            recv_buffer = tensor
        else:
            recv_buffer = torch.empty_like(
                tensor, memory_format=torch.contiguous_format
            )
            self._recv_copy_back.append((tensor, recv_buffer))
        self._p2p_ops.append(dist.P2POp(dist.irecv, recv_buffer, src))
        comm._record_recv_bytes(src, tensor.nbytes)

    def has_ops(self) -> bool:
        return len(self._p2p_ops) > 0

    def wait(self) -> None:
        """Issue the batched ops, wait for completion, and copy-back recvs."""
        if not self._p2p_ops:
            return
        reqs = dist.batch_isend_irecv(self._p2p_ops)
        for req in reqs:
            req.wait()
        for dst_tensor, recv_buffer in self._recv_copy_back:
            dst_tensor.data.copy_(recv_buffer)

        self._p2p_ops.clear()
        self._send_buffers.clear()
        self._recv_copy_back.clear()


class Communicator:
    """Communicate with other ranks.

    1. support pre-build nccl connection by set fake_transfer=True
    2. statistics the communication bytes
    3. support p2p communication within self rank(by self-copy)
    4. support batched p2p with internal shadow buffers via :meth:`batch_p2p`
    """

    def __init__(
        self, _rank: int, send_fn: Callable, recv_fn: Callable, broadcast_fn: Callable
    ):
        self._rank = _rank
        self.send_fn = send_fn
        self.recv_fn = recv_fn
        self.broadcast_fn = broadcast_fn

        # Store NCCL connection information
        self._build_nccl_connection_only = False
        self._p2p_connections: set[int] = set()
        self._collective_connections: set[torch.distributed.ProcessGroup] = set()
        self.dummy_tensor = torch.empty(1, device=torch.cuda.current_device())

        # Record communication bytes
        self._communication_bytes_send = defaultdict(int)
        self._communication_bytes_recv = defaultdict(int)
        self._communication_bytes = 0
        self._group_to_ranks: dict[torch.distributed.ProcessGroup, list[int]] = {
            None: [i for i in range(torch.distributed.get_world_size())]
        }

        # Self copy communication
        self._self_copy_buffer: queue.Queue[torch.Tensor] = queue.Queue()

    def _send_self(self, tensor: torch.Tensor):
        self._self_copy_buffer.put(tensor)

    def _recv_self(self, tensor: torch.Tensor):
        self_copy_tensor = self._self_copy_buffer.get()
        assert self_copy_tensor.storage().size() > 0
        tensor.data.copy_(self_copy_tensor)

    def _build_p2p_connection(self, rank, fn: Callable):
        if rank in self._p2p_connections or rank == self._rank:
            return
        fn(self.dummy_tensor, rank)
        self._p2p_connections.add(rank)

    def _build_collective_connection(self, group, fn: Callable):
        if group in self._collective_connections:
            return
        fn(self.dummy_tensor, group=group)
        self._collective_connections.add(group)

    def set_fake_transfer(self, fake_transfer: bool):
        self._build_nccl_connection_only = fake_transfer

    @staticmethod
    def _is_cuda(tensor: torch.Tensor) -> bool:
        return tensor.device.type == "cuda"

    def _record_send_bytes(self, dst: int, nbytes: int) -> None:
        self._communication_bytes += nbytes
        self._communication_bytes_send[dst] += nbytes

    def _record_recv_bytes(self, src: int, nbytes: int) -> None:
        self._communication_bytes += nbytes
        self._communication_bytes_recv[src] += nbytes

    def batch_p2p(self) -> BatchP2P:
        """Create a fresh :class:`BatchP2P` bound to this communicator."""
        return BatchP2P(self)

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
        *args,
        **kwargs,
    ):
        if self._build_nccl_connection_only:
            return self._build_p2p_connection(dst, self.send_fn)
        if dst == self._rank:
            return self._send_self(tensor.clone().detach().contiguous())

        # NCCL only moves CUDA tensors. A non-CUDA (e.g. CPU-offloaded optimizer)
        # state is staged through a GPU bounce buffer before the send.
        if not self._is_cuda(tensor):
            tensor = tensor.to(torch.cuda.current_device(), non_blocking=True)
        self.send_fn(tensor.contiguous(), dst=dst, *args, **kwargs)

        self._record_send_bytes(dst, tensor.nbytes)

    def recv(
        self,
        tensor: torch.Tensor,
        src,
        *args,
        **kwargs,
    ):
        if self._build_nccl_connection_only:
            return self._build_p2p_connection(src, self.recv_fn)
        if src == self._rank:
            return self._recv_self(tensor)

        # NCCL only moves CUDA tensors. For a non-CUDA destination, receive into a
        # GPU bounce buffer and copy back (this also covers the non-contiguous case
        # since the bounce buffer is contiguous).
        if not self._is_cuda(tensor):
            recv_on_gpu = torch.empty(
                tensor.shape, dtype=tensor.dtype, device=torch.cuda.current_device()
            )
            self.recv_fn(recv_on_gpu, src=src, *args, **kwargs)
            tensor.data.copy_(recv_on_gpu)
        elif not tensor.is_contiguous():
            recv_tensor_contiguous = torch.empty_like(tensor)
            self.recv_fn(recv_tensor_contiguous, src=src, *args, **kwargs)
            tensor.data.copy_(recv_tensor_contiguous)
        else:
            self.recv_fn(tensor, src=src, *args, **kwargs)

        self._record_recv_bytes(src, tensor.nbytes)

    def broadcast(
        self,
        tensor: torch.Tensor,
        *args,
        **kwargs,
    ):
        group = kwargs.get("group")
        if self._build_nccl_connection_only:
            return self._build_collective_connection(group, self.broadcast_fn)

        # NCCL broadcast needs a CUDA tensor; stage CPU tensors through GPU and
        # copy the result back (every rank in the group does the same).
        if not self._is_cuda(tensor):
            gpu_tensor = tensor.to(torch.cuda.current_device(), non_blocking=True)
            self.broadcast_fn(gpu_tensor, *args, **kwargs)
            tensor.data.copy_(gpu_tensor)
        else:
            self.broadcast_fn(tensor, *args, **kwargs)

        if self._group_to_ranks.get(group) is None:
            self._group_to_ranks[group] = torch.distributed.get_process_group_ranks(
                group
            )
        group_ranks = self._group_to_ranks[group]
        if self._rank in group_ranks:
            self._communication_bytes += tensor.nbytes

    def get_communication_bytes(self) -> CommunicationBytes:
        """Get the communication bytes(GB) and reset the communication bytes to 0"""
        communication_bytes = self._communication_bytes
        communication_bytes_send = self._communication_bytes_send
        communication_bytes_recv = self._communication_bytes_recv
        self._communication_bytes = 0
        self._communication_bytes_send = defaultdict(int)
        self._communication_bytes_recv = defaultdict(int)
        return CommunicationBytes(
            communication_bytes_send, communication_bytes_recv, communication_bytes
        )
