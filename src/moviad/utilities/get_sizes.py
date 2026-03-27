from calflops import calculate_flops
import torch

def get_torch_model_size(model:torch.nn.Module) -> float:

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2

def get_model_macs(model:torch.nn.Module, input_shape = (1,3,224,224)) -> int:

    flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=False,
                                      output_precision=4)

    return macs, params

def get_tensor_size(tensor:torch.Tensor) -> float:

    return tensor.numel() * tensor.element_size() / 1024**2

def count_params(module_or_tensor):
    if module_or_tensor is None:
        return 0
    if isinstance(module_or_tensor, torch.nn.Module):
        return sum(p.numel() for p in module_or_tensor.parameters())
    if isinstance(module_or_tensor, torch.Tensor):
        return module_or_tensor.numel()
    if isinstance(module_or_tensor, int):
        return 0
    return 0

def params_to_mb(params):
    bytes_per_param = 4
    return (params * bytes_per_param) / (1024 ** 2)
