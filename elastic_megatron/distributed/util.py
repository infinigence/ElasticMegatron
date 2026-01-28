import inspect
import torch


def has_group_param(obj):
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return False
    for p in sig.parameters.values():
        if p.name == "group":
            return True
    return False


def iter_callables(mod):
    for name in dir(mod):
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if callable(obj):
            yield f"{mod.__name__}.{name}", obj


def get_all_dist_functions_with_group():
    modules = [
        torch.distributed,
        # torch.distributed.distributed_c10d,
    ]
    results = []
    seen = set()
    for mod in modules:
        if mod is None:
            continue
        for qualname, obj in iter_callables(mod):
            if (qualname, id(obj)) in seen:
                continue
            seen.add((qualname, id(obj)))
            if has_group_param(obj):
                module_name = (
                    qualname.split(".")[0] + "." + qualname.split(".")[1]
                )  # torch.distributed
                func_name = qualname.split(".")[2]
                results.append((module_name, func_name, obj))
    return results
