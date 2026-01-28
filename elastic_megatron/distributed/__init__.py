from .dist_patch import patch_torch_distributed

patch_torch_distributed()


from .elastic_process_group import (  # noqa: E402
    ElasticProcessGroup,
    nccl_group_recreate,
    mpu_patch,
    global_barrier_by_gloo,
)

mpu_patch()

__all__ = [
    "nccl_group_recreate",
    "ElasticProcessGroup",
    "global_barrier_by_gloo",
]
