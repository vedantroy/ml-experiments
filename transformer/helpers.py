from torch import Tensor


def assert_shape(tensor_or_tensors, expected_shape):
    if not isinstance(expected_shape, tuple):
        raise ValueError(f"Shape was of type: {type(expected_shape)}, instead of tuple")

    tensors = None
    if isinstance(tensor_or_tensors, Tensor):
        tensors = [tensor_or_tensors]
    else:
        assert isinstance(tensor_or_tensors, list) or isinstance(
            tensor_or_tensors, tuple
        )
        tensors = tensor_or_tensors

    for tensor in tensors:
        if tensor.shape != expected_shape:
            raise ValueError(
                f"(actual) {tuple(tensor.shape)} != (correct) {expected_shape}"
            )
