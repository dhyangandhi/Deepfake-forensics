
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download

print("Downloading/loading checkpoint...")

model_path = hf_hub_download(
    repo_id="Yashikaysn29/deepshield",
    filename="best_model.pth"
)

state_dict = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

print("\n========== CHECKPOINT INSPECTION ==========")
print("Checkpoint:", model_path)
print("File size:", round(Path(model_path).stat().st_size / (1024 * 1024), 2), "MB")
print("Number of parameters:", len(state_dict))

print("\nFirst 15 parameter keys:")

for key in list(state_dict.keys())[:15]:
    print(key, state_dict[key].shape)

print("\nLast 10 parameter keys:")

for key in list(state_dict.keys())[-10:]:
    print(key, state_dict[key].shape)

print("\n===========================================")
