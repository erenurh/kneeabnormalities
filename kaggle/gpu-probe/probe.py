import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(), "| n_gpu:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
