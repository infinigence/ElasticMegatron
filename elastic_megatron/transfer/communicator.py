import os
import queue
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from .chunk_schedule import chunk_ranges, slice_spans


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

        nbytes = tensor.nbytes
        # NCCL only moves CUDA tensors; stage a non-CUDA (e.g. CPU-offloaded
        # optimizer) tensor through the GPU before sending.
        if not comm._is_cuda(tensor):
            tensor = tensor.to(torch.cuda.current_device(), non_blocking=True)
        send_buffer = tensor.contiguous()
        self._send_buffers.append(send_buffer)
        self._p2p_ops.append(dist.P2POp(dist.isend, send_buffer, dst))
        comm._record_send_bytes(dst, nbytes)

    def irecv(self, tensor: torch.Tensor, src: int) -> None:
        comm = self._communicator
        if comm._build_nccl_connection_only:
            comm._build_p2p_connection(src, comm.recv_fn)
            return
        if src == comm._rank:
            comm._recv_self(tensor)
            return

        # NCCL only writes CUDA tensors; a non-CUDA (CPU-offloaded) or
        # non-contiguous destination receives into a contiguous GPU buffer that
        # is copied back to ``tensor`` (staging out) on wait().
        if comm._is_cuda(tensor) and tensor.is_contiguous():
            recv_buffer = tensor
        else:
            recv_buffer = torch.empty(
                tensor.shape, dtype=tensor.dtype, device=torch.cuda.current_device()
            )
            self._recv_copy_back.append((tensor, recv_buffer))
        self._p2p_ops.append(dist.P2POp(dist.irecv, recv_buffer, src))
        comm._record_recv_bytes(src, tensor.nbytes)

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
    4. accumulate raw p2p ops via :meth:`batch_p2p`

    This is the per-op primitive layer (send/recv/broadcast + staging + self-copy
    + connection building + byte accounting). The higher-level packed / bucketed
    cross-rank exchange lives in :class:`BatchedTransfer`, which composes this.
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
        """Return the per-peer and total communication byte counts (raw bytes,
        not GB) and reset the counters to 0."""
        communication_bytes = self._communication_bytes
        communication_bytes_send = self._communication_bytes_send
        communication_bytes_recv = self._communication_bytes_recv
        self._communication_bytes = 0
        self._communication_bytes_send = defaultdict(int)
        self._communication_bytes_recv = defaultdict(int)
        return CommunicationBytes(
            communication_bytes_send, communication_bytes_recv, communication_bytes
        )


class BatchedTransfer:
    """Batched, packed, deadlock-safe cross-rank exchange built on a
    :class:`Communicator`.

    :meth:`transfer` is the all-in-one transport entry point (used by _pre/_post
    DP gather-scatter and the no-pack legacy). The packed path coalesces each
    peer's slices through a pair of REUSED uint8 GPU staging buffers, split into
    byte chunks of at most ``max_inflight_bytes`` — so staging residency is
    bounded by ~2x the cap regardless of model size. Chunk boundaries are derived
    independently on both ranks of a pair from the same ordered slice list (see
    chunk_schedule.py), so the packed path needs no collective exchange.

    For the reshard main process, the same exchange is exposed split into
    :meth:`pack` / :meth:`exchange` / :meth:`unpack` so the caller can release a
    chunk's src after pack and create its dst after exchange (flat-peak
    approach-2); see those methods.
    """

    def __init__(self, communicator: "Communicator"):
        self._comm = communicator
        # Per-transfer phase timings (ms), populated when collect_phase_ms=True:
        # pack (H2D stage-in), comm (NCCL), unpack (D2H stage-out). Input data
        # for the next-phase overlap design (docs/buffer_opt/cpu-adam-overlap.md).
        self.last_phase_ms: dict[str, float] | None = None
        # cpu-adam D2H (unpack stage-out) is the reshard bottleneck (~2.9 GB/s):
        # HDO rebinds the fp32 master to PAGEABLE host memory, so GPU->pageable
        # is effectively synchronous. Routing the bulk D2H through a reused PINNED
        # host bounce (~25 GB/s) + a CPU scatter into the pageable dst recovers
        # most of that. Local-only (same bytes/NCCL ops; no cross-rank match
        # needed), no-op for GPU-adam / already-pinned dst. Default OFF until the
        # 30B A/B confirms; see docs/buffer_opt/cpu-adam-overlap.md.
        self._pinned_bounce = os.getenv("ELASTIC_TRANSFER_PINNED_BOUNCE", "0") == "1"
        self._bounce_buf: torch.Tensor | None = None
        # Cap the PINNED bounce buffer: the D2H is looped through it in <=cap-byte
        # sub-chunks, so the page-locked footprint stays small REGARDLESS of the
        # (large) chunk size. A multi-GB pinned alloc is a slow, synchronizing
        # cudaHostAlloc and competes with NCCL's own pinned transport pool;
        # bounding it keeps the pinned-bounce D2H win without that contention.
        # <=0 => unbounded (one buffer the full chunk size; for A/B).
        self._bounce_cap = int(os.getenv("ELASTIC_TRANSFER_PINNED_BOUNCE_BYTES", str(256 << 20)))

    def transfer(
        self,
        send_tasks: Dict[int, List[torch.Tensor]],
        recv_tasks: Dict[int, List[torch.Tensor]],
        *,
        pack: bool = True,
        max_inflight_bytes: int | None = None,
        collect_phase_ms: bool = False,
    ) -> None:
        """Move a batch of cross-rank tensor slices.

        ``send_tasks[peer]`` / ``recv_tasks[peer]`` are ordered lists of the
        slices to send to / receive from ``peer`` (recv entries are contiguous
        landing buffers; send entries may be strided views). Self-rank copies
        are issued by the caller via ``Communicator.send`` / ``recv``.

        With ``pack``, each peer pair exchanges in rounds: every round packs at
        most ``max_inflight_bytes`` bytes per direction into a reused GPU
        staging buffer and issues one ``batch_isend_irecv``. The pack/unpack
        byte copies double as CPU<->GPU staging for offloaded (HDO) state.
        ``max_inflight_bytes`` must be identical on every rank (both ends must
        derive the same chunk count); None/<=0 => single chunk per peer
        (legacy residency).
        A smaller cap lowers staging residency but serialises the exchange into
        more rounds (one ``batch_isend_irecv`` + wait per round) — latency
        rises roughly with ceil(bytes/cap); pick the cap to fit the memory
        budget, not smaller. Without ``pack`` each slice is a separate,
        individually staged p2p op in one whole-exchange batch
        (``max_inflight_bytes`` is ignored on this path; legacy debug fallback).
        """
        comm = self._comm
        # In fake_transfer mode the caller's collectors have already built the
        # NCCL connections and pass empty task dicts; nothing to move.
        if comm._build_nccl_connection_only:
            return

        self.last_phase_ms = (
            {"pack": 0.0, "comm": 0.0, "unpack": 0.0} if collect_phase_ms else None
        )

        world_size = torch.distributed.get_world_size()
        # The XOR "butterfly" pairs ranks step by step; both ranks of a pair
        # meet at the same step. num_steps rounds up to the next power of two;
        # out-of-range peers are skipped — every real pair still meets once.
        num_steps = 1 << ((world_size - 1).bit_length())

        if not pack:
            # One batch_isend_irecv per butterfly STEP (peer pair), not a single
            # batch over all peers. This path emits one p2p op per slice, and
            # coalescing every peer's slices into a single batch_isend_irecv
            # deadlocks when the reshard plan gives ranks an asymmetric op set
            # (dense multi-strategy reshards): the ranks diverge on the NCCL
            # collective sequence (a per-rank SeqNum split, caught at the wait).
            # One batch per paired step keeps each coalesced group to a single
            # symmetric peer pair — the structure the packed path below uses.
            # The cap is ignored on this path (legacy per-slice residency).
            for step in range(1, num_steps):
                peer = comm._rank ^ step
                if peer >= world_size:
                    continue
                peer_sends = send_tasks.get(peer)
                peer_recvs = recv_tasks.get(peer)
                if not peer_sends and not peer_recvs:
                    continue
                batch = comm.batch_p2p()
                self._enqueue_unpacked(batch, peer, peer_sends, peer_recvs)
                with self._phase("comm"):
                    batch.wait()
            return

        # Reused staging buffers, sized lazily to the largest chunk seen and
        # dropped at the end of the transfer (freed back to the cached pool;
        # the window-end empty_cache in elastic_manager.reshard reclaims them).
        stage_bufs: dict[str, torch.Tensor] = {}

        def stage(kind: str, nbytes: int) -> torch.Tensor:
            buf = stage_bufs.get(kind)
            if buf is None or buf.numel() < nbytes:
                stage_bufs[kind] = buf = torch.empty(
                    nbytes, dtype=torch.uint8, device=torch.cuda.current_device()
                )
            return buf[:nbytes]

        for step in range(1, num_steps):
            peer = comm._rank ^ step
            if peer >= world_size:
                continue
            peer_sends = send_tasks.get(peer) or []
            peer_recvs = recv_tasks.get(peer) or []
            if not peer_sends and not peer_recvs:
                continue

            # Byte views once per peer. .contiguous() may materialise a strided
            # send view (same cost as the old whole-buffer pack); recv landings are
            # contiguous by contract, so their views are zero-copy.
            send_bytes = [
                t.contiguous().view(torch.uint8).reshape(-1) for t in peer_sends
            ]
            recv_bytes = [t.view(torch.uint8).reshape(-1) for t in peer_recvs]
            send_sizes = [b.numel() for b in send_bytes]
            recv_sizes = [b.numel() for b in recv_bytes]

            send_chunks = chunk_ranges(sum(send_sizes), max_inflight_bytes)
            recv_chunks = chunk_ranges(sum(recv_sizes), max_inflight_bytes)
            # A's send list to B IS B's landing list from A (same slices, same
            # order), so both ends compute identical chunk counts per direction
            # — the round count below agrees without any exchange.
            for r in range(max(len(send_chunks), len(recv_chunks))):
                s_chunk = send_chunks[r] if r < len(send_chunks) else None
                r_chunk = recv_chunks[r] if r < len(recv_chunks) else None
                s_stage = r_stage = None
                if s_chunk is not None:
                    s_stage = stage("send", s_chunk[1] - s_chunk[0])
                    with self._phase("pack"):
                        self._pack_chunk(send_bytes, send_sizes, s_chunk, s_stage)
                if r_chunk is not None:
                    r_stage = stage("recv", r_chunk[1] - r_chunk[0])

                batch = comm.batch_p2p()
                # Lower-ranked peer enqueues sends first, higher-ranked recvs
                # first; batch_isend_irecv preserves enqueue order — no deadlock.
                if comm._rank < peer:
                    if s_stage is not None:
                        batch.isend(s_stage, dst=peer)
                    if r_stage is not None:
                        batch.irecv(r_stage, src=peer)
                else:
                    if r_stage is not None:
                        batch.irecv(r_stage, src=peer)
                    if s_stage is not None:
                        batch.isend(s_stage, dst=peer)
                with self._phase("comm"):
                    batch.wait()
                # wait() orders the default stream after the NCCL ops, so both
                # the unpack below and the next round's repack into the same
                # staging buffer are stream-ordered — reuse is safe without a
                # host sync.
                if r_chunk is not None:
                    with self._phase("unpack"):
                        self._unpack_chunk(recv_bytes, recv_sizes, r_chunk, r_stage)

    # ------------------------------------------------------------------------
    # Phase-split entry points (pack / exchange / unpack)
    #
    # The single :meth:`transfer` above reads src and writes dst inside one call,
    # so the caller must have both endpoints live across it. The reshard
    # main-process flat-peak path needs to RELEASE a chunk's src before the comm
    # and CREATE its dst after, so a chunk's src and dst never coexist. These
    # three methods split :meth:`transfer` at exactly those two seams:
    #
    #   staged   = pack(send_tasks)        # reads src  -> caller releases src
    #   received = exchange(staged, nbytes)# staging only (src gone, dst absent)
    #   ...                                # caller creates dst
    #   unpack(received, recv_tasks)       # writes dst
    #
    # ``exchange`` keeps the butterfly pairing / ordering of :meth:`transfer`
    # verbatim — the deadlock-critical machinery is untouched — and touches ONLY
    # the staging byte-buffers (the chain src -> pack -> staging -> [NCCL] ->
    # staging -> unpack -> dst means the comm never references src/dst). They are
    # additive: ``transfer`` is unchanged and still serves _pre/_post and the
    # no-pack legacy. ``max_inflight_bytes`` is consumed by the caller's CHUNK
    # granularity here, so a single ``batch_isend_irecv`` per peer carries the
    # whole (cap-sized) chunk; the within-peer cap-loop is not re-run.
    # ------------------------------------------------------------------------

    def pack(
        self, send_tasks: Dict[int, List[torch.Tensor]]
    ) -> Dict[int, torch.Tensor]:
        """Pack each peer's send slices into one contiguous uint8 GPU staging
        buffer. Returns ``{peer: send_staging}``.

        After ``pack`` returns, the source optimizer tensors may be released —
        their bytes now live in the staging buffers (the pack copy doubles as the
        CPU->GPU stage-in for offloaded state, exactly as in :meth:`transfer`).
        Self-rank sends are not here; the caller routes them through the
        communicator self-copy queue. A no-op in fake_transfer mode (the caller's
        collectors built the connections; nothing to move)."""
        if self._comm._build_nccl_connection_only:
            return {}
        staged: Dict[int, torch.Tensor] = {}
        for peer, slices in send_tasks.items():
            if not slices:
                continue
            send_bytes = [t.contiguous().view(torch.uint8).reshape(-1) for t in slices]
            sizes = [b.numel() for b in send_bytes]
            total = sum(sizes)
            if total == 0:
                continue
            buf = torch.empty(
                total, dtype=torch.uint8, device=torch.cuda.current_device()
            )
            with self._phase("pack"):
                self._pack_chunk(send_bytes, sizes, (0, total), buf)
            staged[peer] = buf
        return staged

    def exchange(
        self,
        send_staging: Dict[int, torch.Tensor],
        recv_nbytes: Dict[int, int],
    ) -> Dict[int, torch.Tensor]:
        """Butterfly-exchange pre-packed staging buffers with every peer; return
        ``{peer: recv_staging}``.

        ``recv_nbytes[peer]`` is the number of bytes to receive from ``peer``,
        sized by the caller from dst placeholder metadata (NOT the dst tensors,
        which may not exist yet) so the dst allocation can stay AFTER this call.
        One recv staging buffer is allocated per peer; one ``batch_isend_irecv``
        per pair, with the same peer pairing and send/recv ordering as
        :meth:`transfer` (lower rank sends first) — deadlock-safe. Touches ONLY
        staging buffers."""
        comm = self._comm
        if comm._build_nccl_connection_only:
            return {}
        world_size = torch.distributed.get_world_size()
        num_steps = 1 << ((world_size - 1).bit_length())
        received: Dict[int, torch.Tensor] = {}
        for step in range(1, num_steps):
            peer = comm._rank ^ step
            if peer >= world_size:
                continue
            s_stage = send_staging.get(peer)
            has_send = s_stage is not None and s_stage.numel() > 0
            r_bytes = recv_nbytes.get(peer, 0)
            if not has_send and r_bytes <= 0:
                continue
            r_stage = None
            if r_bytes > 0:
                r_stage = torch.empty(
                    r_bytes, dtype=torch.uint8, device=torch.cuda.current_device()
                )
                received[peer] = r_stage
            batch = comm.batch_p2p()
            # Lower-ranked peer enqueues its send first, higher-ranked its recv
            # first; batch_isend_irecv preserves enqueue order — no deadlock.
            if comm._rank < peer:
                if has_send:
                    batch.isend(s_stage, dst=peer)
                if r_stage is not None:
                    batch.irecv(r_stage, src=peer)
            else:
                if r_stage is not None:
                    batch.irecv(r_stage, src=peer)
                if has_send:
                    batch.isend(s_stage, dst=peer)
            with self._phase("comm"):
                batch.wait()
        return received

    def unpack(
        self,
        received: Dict[int, torch.Tensor],
        recv_tasks: Dict[int, List[torch.Tensor]],
    ) -> None:
        """Scatter each peer's recv staging buffer back into the destination
        landing slices (dst now created). The byte copy doubles as the GPU->CPU
        stage-out for offloaded dst state. Slice order matches the sender's
        :meth:`pack`, so the bytes line up. A no-op in fake_transfer mode."""
        if self._comm._build_nccl_connection_only:
            return
        for peer, slices in recv_tasks.items():
            if not slices:
                continue
            r_stage = received.get(peer)
            if r_stage is None:
                continue
            recv_bytes = [t.view(torch.uint8).reshape(-1) for t in slices]
            sizes = [b.numel() for b in recv_bytes]
            total = sum(sizes)
            if total == 0:
                continue
            assert r_stage.numel() == total, (
                f"recv staging for peer {peer} is {r_stage.numel()} bytes but the "
                f"landing slices need {total}; the recv_nbytes metadata disagrees "
                "with the dst layout (see _chunk_recv_nbytes)."
            )
            with self._phase("unpack"):
                if self._needs_bounce(slices):
                    # D2H GPU->PINNED (fast ~25 GB/s; synchronous so the bounce is
                    # filled before the CPU scatter reads it), then scatter
                    # PINNED->pageable dst slices (host memcpy). Replaces the
                    # per-slice GPU->pageable D2H (~2.9 GB/s) that dominated the
                    # cpu-adam reshard. Looped in <=_bounce_cap sub-chunks so the
                    # page-locked buffer stays small. Same bytes -> bit-exact.
                    cap = self._bounce_cap
                    step = min(cap, total) if cap and cap > 0 else total
                    bounce = self._pinned_bounce_buf(step)
                    off = 0
                    while off < total:
                        n = min(step, total - off)
                        sub = bounce[:n]
                        sub.copy_(r_stage[off : off + n])
                        self._unpack_chunk(recv_bytes, sizes, (off, off + n), sub)
                        off += n
                else:
                    self._unpack_chunk(recv_bytes, sizes, (0, total), r_stage)

    def _needs_bounce(self, slices: List[torch.Tensor]) -> bool:
        """True when the pinned-bounce D2H path applies: enabled, and the landing
        slices are PAGEABLE host (cpu-adam offloaded state). GPU dst (GPU-adam)
        and already-pinned host dst both skip it -- their direct copy is already
        fast. Representative device probe on the first slice (a peer's slices all
        belong to one optimizer state, so share a device/pinning)."""
        if not self._pinned_bounce or not slices:
            return False
        first = slices[0]
        return first.device.type == "cpu" and not first.is_pinned()

    def _pinned_bounce_buf(self, nbytes: int) -> torch.Tensor:
        """A single reused PINNED host uint8 bounce buffer, grown to the largest
        chunk seen (pinned allocation is costly, so reuse). Sliced to ``nbytes``.
        Reuse is safe because the D2H into it is synchronous and the CPU scatter
        out of it completes before the next peer/chunk repacks it."""
        buf = self._bounce_buf
        if buf is None or buf.numel() < nbytes:
            self._bounce_buf = buf = torch.empty(
                nbytes, dtype=torch.uint8, pin_memory=True
            )
        return buf[:nbytes]

    def _enqueue_unpacked(
        self,
        batch: BatchP2P,
        peer: int,
        peer_sends: List[torch.Tensor] | None,
        peer_recvs: List[torch.Tensor] | None,
    ) -> None:
        if not peer_sends and not peer_recvs:
            return
        if self._comm._rank < peer:
            for tensor in peer_sends or []:
                batch.isend(tensor, dst=peer)
            for tensor in peer_recvs or []:
                batch.irecv(tensor, src=peer)
        else:
            for tensor in peer_recvs or []:
                batch.irecv(tensor, src=peer)
            for tensor in peer_sends or []:
                batch.isend(tensor, dst=peer)

    @contextmanager
    def _phase(self, name: str):
        """Accumulate the wrapped block's duration into last_phase_ms[name]
        (CUDA-synced). No-op (and no synchronize) when timing is off."""
        if self.last_phase_ms is None:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.last_phase_ms[name] += (time.perf_counter_ns() - t0) / 1e6

    @staticmethod
    def _pack_chunk(
        byte_views: List[torch.Tensor],
        sizes: List[int],
        chunk: Tuple[int, int],
        stage: torch.Tensor,
    ) -> None:
        """Copy byte range ``chunk`` of the logical slice concatenation into
        ``stage``. A CPU (offloaded) source slice is staged to GPU by the
        cross-device copy itself, ordered before the NCCL ops by the shared
        default stream. NB: offloaded (HDO) optimizer state is mostly pageable
        CPU memory, so non_blocking is effectively synchronous for it; correct,
        just not overlapped."""
        offset = 0
        for idx, t_off, n in slice_spans(sizes, *chunk):
            stage[offset : offset + n].copy_(
                byte_views[idx][t_off : t_off + n], non_blocking=True
            )
            offset += n

    @staticmethod
    def _unpack_chunk(
        byte_views: List[torch.Tensor],
        sizes: List[int],
        chunk: Tuple[int, int],
        stage: torch.Tensor,
    ) -> None:
        """Scatter received ``stage`` bytes back into the destination slices.
        CPU destinations are staged out by the copy (blocking, as before —
        the host reads offloaded state right after transfer)."""
        offset = 0
        for idx, t_off, n in slice_spans(sizes, *chunk):
            byte_views[idx][t_off : t_off + n].copy_(stage[offset : offset + n])
            offset += n
