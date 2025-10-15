import os
import torch

def inspect_pt_files(folder='/workspace/ba_inputs'):
    pt_files = [f for f in os.listdir(folder) if f.endswith('.pt') or f.endswith('.pth')]
    if not pt_files:
        print("No .pt or .pth files found in the folder.")
        return

    for fname in pt_files:
        print("="*80)
        print(f"File: {fname}")
        try:
            data = torch.load(os.path.join(folder, fname), map_location="cpu")
            if isinstance(data, dict):
                for k, v in data.items():
                    if torch.is_tensor(v):
                        print(f"  {k}: tensor, shape {tuple(v.shape)}, dtype {v.dtype}")
                    elif isinstance(v, (int, float, str)):
                        print(f"  {k}: {type(v).__name__}, value: {v}")
                    elif isinstance(v, dict):
                        print(f"  {k}: dict (nested)")
                    elif isinstance(v, list):
                        print(f"  {k}: list, length {len(v)}")
                    else:
                        print(f"  {k}: {type(v)}")
            elif torch.is_tensor(data):
                print(f"  (no dict) tensor, shape {tuple(data.shape)}, dtype {data.dtype}")
            else:
                print("  File content is not a dict or tensor.")
        except Exception as e:
            print(f"  Failed to load/inspect: {e}")
    print("="*80)

if __name__ == "__main__":
    inspect_pt_files()
