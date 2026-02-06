import torch


class DataParallelPassthrough(torch.nn.DataParallel):
    """DataParallel wrapper that forwards missing attributes to the wrapped module.

    This allows calling custom helper methods (e.g., train_step) while keeping
    DataParallel's forward for multi-GPU execution.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.module, name)
            if callable(attr) and hasattr(attr, "__func__"):
                return lambda *args, **kwargs: attr.__func__(self, *args, **kwargs)
            return attr


def wrap_model_for_dataparallel(model, device_ids=None):
    if device_ids is None:
        return DataParallelPassthrough(model)
    return DataParallelPassthrough(model, device_ids=device_ids)
