import queue
from collections.abc import Callable

import torch


class Communicator:
    """Communicate with other ranks.

    1. support pre-build nccl connection by set fake_transfer=True
    2. statistics the communication bytes
    3. support p2p communication within self rank(by self-copy)
    4. TODO: support high performance communication by use buffer
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

        self._communication_bytes += tensor.nbytes

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

        self._communication_bytes += tensor.nbytes

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

    def get_communication_bytes(self) -> float:
        """Get the communication bytes(GB) and reset the communication bytes to 0"""
        communication_bytes = self._communication_bytes
        self._communication_bytes = 0
        return communication_bytes / (1024**3)
