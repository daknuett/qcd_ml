import torch
import qcd_ml.util.qcd
import qcd_ml.util.solver


def get_device_by_reference(reference: torch.Tensor):
    """
    This is a utility function that gives a result that can be passed
    to ``torch.Tensor.to()``. It takes a reference tensor and allows
    the user to put a tensor on the same device as the reference tensor.
    """
    device = reference.get_device()

    if device == -1:
        return "cpu"
    return f"cuda:{device}"
