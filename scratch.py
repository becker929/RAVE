import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

print(model)

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:
    with record_function("model_inference"):
        model(inputs)
print(prof.key_averages().table(sort_by="gpu_time_total", row_limit=10))
prof.export_stacks("/content/profiler_cpu_stacks.txt", "self_cpu_time_total")
prof.export_stacks("/content/profiler_gpu_stacks.txt", "self_gpu_time_total")


# model = models.resnet18().cuda()
# inputs = torch.randn(5, 3, 224, 224).cuda()

# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
# ) as prof:
#     with record_function("model_inference"):
#         model(inputs)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
