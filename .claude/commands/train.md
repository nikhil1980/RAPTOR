Run the RAPTOR training pipeline.

Arguments: $ARGUMENTS

Steps:
1. Check whether `WANDB_PROJECT` is set:
   ```bash
   echo "WANDB_PROJECT=${WANDB_PROJECT:-<not set>}"
   ```
   If unset, warn the user to run `export WANDB_PROJECT=<your-project-name>` before proceeding.

2. Check GPU availability:
   ```bash
   python3 -c "import torch; avail=torch.cuda.is_available(); n=torch.cuda.device_count(); mem=[round(torch.cuda.get_device_properties(i).total_memory/1e9,1) for i in range(n)]; print(f'CUDA: {avail} | Devices: {n} | VRAM: {mem} GB')"
   ```
   Warn if no GPU is detected — CPU training will be impractically slow.

3. Run training (pass any user-provided extra flags from $ARGUMENTS):
   ```bash
   python train/train_dinov3_rtdetr_ov.py --config-file config.json $ARGUMENTS
   ```

4. After training completes, report the checkpoint:
   ```bash
   python3 -c "
   import os, glob
   d = 'runs/dinov3_rtdetr/final'
   if os.path.exists(d):
       files = [f for f in glob.glob(d+'/**/*', recursive=True) if os.path.isfile(f)]
       total = sum(os.path.getsize(f) for f in files)
       print(f'Checkpoint: {d}/  ({total/1e6:.0f} MB, {len(files)} files)')
   else:
       print('WARNING: No checkpoint found at runs/dinov3_rtdetr/final/')
   "
   ```
