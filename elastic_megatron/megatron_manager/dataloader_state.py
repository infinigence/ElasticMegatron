from collections.abc import Callable

import torch


class DataloaderState:
    """Responsible for building and caching the datasets, data loaders, and data iterators.

    In Megatron-LM, the data_iterators are built in the following order: data_sets -> data_loaders -> data_iterators

    Rebuild data_sets is too heavy and would cost too much time, so we need to cache the datasets and reuse them when resharding.

    Key Points:
    1. Patch the func of build_train_valid_test_datasets()
        1.1 In DataloaderState.register(), save the argument of build_train_valid_test_datasets()
        1.2 Cache the datasets built by build_train_valid_test_datasets()

    2. Patch the func of build_train_valid_test_data_loaders() in rebuild process
        2.1 No need to build datasets in build_train_valid_test_data_loaders()
        2.2 Ignore the broadcast

    3. Rebuild process
        3.1 If not is_dataset_built_on_rank(), return [None] * 3
        3.2 If datasets are not built, build them by build_train_valid_test_data_iterators()
        3.3 Patch the func of build_train_valid_test_data_loaders()
        3.4 Call build_train_valid_test_data_iterators() to build data iterators
        3.5 Restore the func of build_train_valid_test_data_loaders()
        3.6 Return the data iterators
    """

    train_valid_test_datasets_provider: Callable = None
    build_train_valid_test_datasets: Callable = None

    train_datasets = None
    valid_datasets = None
    test_datasets = None

    @staticmethod
    def is_dataset_built_on_rank():
        from megatron.core import mpu

        return (
            mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
        ) and mpu.get_tensor_model_parallel_rank() == 0

    @classmethod
    def register(cls, train_valid_test_datasets_provider: Callable):
        cls.train_valid_test_datasets_provider = train_valid_test_datasets_provider

        from megatron.training import training

        cls.origin_build_train_valid_test_datasets = (
            training.build_train_valid_test_datasets
        )
        training.build_train_valid_test_datasets = (
            cls._patched_build_train_valid_test_datasets
        )

    @classmethod
    def _patched_build_train_valid_test_datasets(cls, *args, **kwargs):
        if cls.train_datasets is not None:
            return cls.train_datasets, cls.valid_datasets, cls.test_datasets

        datasets = cls.origin_build_train_valid_test_datasets(*args, **kwargs)
        cls.train_datasets = datasets[0]
        cls.valid_datasets = datasets[1]
        cls.test_datasets = datasets[2]
        torch.cuda.synchronize()
        return datasets

    @classmethod
    def _build_train_valid_test_data_loaders(cls, *args, **kwargs):
        """Ignore the broadcast logic of build_pretraining_data_loader and do not need to build datasets."""
        from megatron.training import get_args

        try:
            # Megatron 0.11 and earlier.
            from megatron.legacy.data.data_samplers import build_pretraining_data_loader
        except ModuleNotFoundError:
            # Megatron 0.16 relocated data_samplers into megatron.training.datasets.
            from megatron.training.datasets.data_samplers import (
                build_pretraining_data_loader,
            )

        from ..resharding.util import get_megatron_version_minor

        args = get_args()
        train_dataloader = build_pretraining_data_loader(
            cls.train_datasets, args.consumed_train_samples
        )
        if args.skip_train:
            valid_dataloader = build_pretraining_data_loader(cls.valid_datasets, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(
                cls.valid_datasets, args.consumed_valid_samples
            )
        test_dataloader = build_pretraining_data_loader(cls.test_datasets, 0)

        # 0.16's build_train_valid_test_data_iterators expects valid_dataloaders
        # to be a list (supports multiple validation sets).
        if get_megatron_version_minor() >= 16:
            valid_dataloader = [valid_dataloader]

        return train_dataloader, valid_dataloader, test_dataloader

    @classmethod
    def _build_train_valid_test_data_iterators(cls):
        # Build datasets
        from megatron.training import training

        if cls.is_dataset_built_on_rank() and cls.train_datasets is None:
            # In build_train_valid_test_datasets(), there are some barriers and broadcasts, we need to ignore them
            origin_barrier = torch.distributed.barrier
            origin_broadcast = torch.distributed.broadcast
            torch.distributed.barrier = lambda *args, **kwargs: None
            torch.distributed.broadcast = lambda *args, **kwargs: None

            training.build_train_valid_test_datasets(
                cls.train_valid_test_datasets_provider
            )

            torch.distributed.barrier = origin_barrier
            torch.distributed.broadcast = origin_broadcast

        # Patch dataloader build function
        origin_build_train_valid_test_data_loaders = (
            training.build_train_valid_test_data_loaders
        )
        training.build_train_valid_test_data_loaders = (
            cls._build_train_valid_test_data_loaders
        )

        # Build data iterators
        iterators = training.build_train_valid_test_data_iterators(
            cls.train_valid_test_datasets_provider
        )

        # Restore dataloader build function
        training.build_train_valid_test_data_loaders = (
            origin_build_train_valid_test_data_loaders
        )
        return iterators

    @classmethod
    def build_iterators(cls):
        from megatron.core.num_microbatches_calculator import (
            reconfigure_num_microbatches_calculator,
        )
        from megatron.training import get_args

        args = get_args()
        consumed_samples = torch.tensor(
            [args.consumed_train_samples],
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        torch.distributed.broadcast(consumed_samples, src=0)
        args.consumed_train_samples = consumed_samples.item()
        reconfigure_num_microbatches_calculator(
            args.rank,
            args.rampup_batch_size,
            args.global_batch_size,
            args.micro_batch_size,
            args.data_parallel_size,
        )

        # Build iterators first; on ranks that don't own the dataset, this returns
        # [None, None, None]. We need to decide do_train/do_valid/do_test based on
        # *whether the corresponding iterator was actually built*, not just on
        # args.eval_iters > 0 —— 否则当 --split 没有 test 分片(如 "98,2,0")时,
        # 全部 rank 都会把 do_test 设成 True,pretrain() 末尾的 final test eval
        # 会拿到 None 的 test_data_iterator,触发 `assert data_iterator is not None`。
        # 这与 Megatron 原始 build_train_valid_test_data_iterators 的语义保持一致。
        if cls.is_dataset_built_on_rank():
            iterators = cls._build_train_valid_test_data_iterators()
        else:
            iterators = [None, None, None]

        train_iter, valid_iter, test_iter = iterators
        do_train = (train_iter is not None) and (args.train_iters > 0)
        do_valid = (valid_iter is not None) and (args.eval_iters > 0)
        do_test = (test_iter is not None) and (args.eval_iters > 0)

        # 各 rank 取并:只要任一 rank 上某 split 有 iterator,全 rank 都置 True
        # (eval / pretrain 后续的 evaluate_* 是 collective,必须每个 rank 都参与)。
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        args.do_train = bool(flags[0].item())
        args.do_valid = bool(flags[1].item())
        args.do_test = bool(flags[2].item())

        return iterators
