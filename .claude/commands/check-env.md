Verify the RAPTOR environment is fully set up for training and inference.

Arguments: $ARGUMENTS

Run each check below and report PASS / WARN / FAIL for each item.

1. **Python version** (need 3.9+):
   ```bash
   python3 --version
   ```

2. **GPU availability**:
   ```bash
   python3 -c "
   import torch
   avail = torch.cuda.is_available()
   n = torch.cuda.device_count()
   names = [torch.cuda.get_device_name(i) for i in range(n)]
   mem = [round(torch.cuda.get_device_properties(i).total_memory/1e9,1) for i in range(n)]
   print(f'CUDA: {avail} | Devices: {n} | Names: {names} | VRAM: {mem} GB')
   "
   ```

3. **Key packages**:
   ```bash
   python3 -c "
   pkgs = ['torch','torchvision','transformers','datasets','accelerate','pycocotools','lvis','wandb','PIL','open_clip','fiftyone']
   missing = []
   for p in pkgs:
       try: __import__(p); print(f'  OK  {p}')
       except ImportError: missing.append(p); print(f'  MISSING  {p}')
   if missing: print(f'\nInstall missing: pip install {\" \".join(missing)}')
   "
   ```

4. **HuggingFace auth** (required for gated DINOv3 weights):
   ```bash
   huggingface-cli whoami
   ```

5. **W&B auth**:
   ```bash
   wandb status
   echo "WANDB_PROJECT=${WANDB_PROJECT:-<not set>}"
   ```

6. **Data readiness**:
   ```bash
   python3 -c "
   import os
   checks = {
     'Train annotations': 'datasets/mixture/annotations/coco_merged/instances_train_merged.json',
     'Val annotations':   'datasets/mixture/annotations/coco_merged/instances_val_merged.json',
     'Train images dir':  'datasets/mixture/images/train2017',
     'Val images dir':    'datasets/mixture/images/val2017',
   }
   for name, path in checks.items():
       print(f'  {\"OK\" if os.path.exists(path) else \"MISSING\":7} {name}: {path}')
   "
   ```

7. **Model checkpoint**:
   ```bash
   python3 -c "import os; d='runs/dinov3_rtdetr/final'; print('  OK     Checkpoint:', d) if os.path.exists(d) else print('  MISSING Checkpoint — run /train to create')"
   ```

8. **Config file**:
   ```bash
   python3 -c "import json; cfg=json.load(open('config.json')); print('  OK     config.json valid —', len(cfg), 'top-level keys')"
   ```

Summarise at the end: list any items that are MISSING or WARN with the remediation step.
