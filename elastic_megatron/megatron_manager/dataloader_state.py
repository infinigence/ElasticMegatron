from typing import Callable
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
        from megatron.legacy.data.data_samplers import build_pretraining_data_loader

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
        from megatron.training import get_args
        from megatron.core.num_microbatches_calculator import (
            reconfigure_num_microbatches_calculator,
        )

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

        # Sync args.do_train/valid/test
        args.do_train = args.train_iters > 0
        args.do_valid = args.eval_iters > 0
        args.do_test = args.eval_iters > 0

        if not cls.is_dataset_built_on_rank():
            return [None] * 3
        return cls._build_train_valid_test_data_iterators()
