from torch.profiler import profile, record_function, ProfilerActivity

def torch_profile(func):
    def wrapper(*args, **kwargs):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True,
                     ) as prof:
            with record_function("wrapped_function"):
                result = func(*args, **kwargs)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
        return result
    return wrapper