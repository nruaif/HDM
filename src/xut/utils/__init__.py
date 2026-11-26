import torch
from .. import env


def isiterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def compile_wrapper(func, **compile_kwargs):
    compiled = None

    def runner(*args, **kwargs):
        nonlocal compiled
        if env.TORCH_COMPILE:
            if compiled is None:
                merged = dict(env.COMPILE_ARGS)
                merged.update(compile_kwargs)
                compiled = torch.compile(func, **merged)
            return compiled(*args, **kwargs)
        return func(*args, **kwargs)

    return runner
