Run the full RAPTOR data download and preparation pipeline (COCO + LVIS + OpenImages-V7).

Arguments: $ARGUMENTS

Steps:
1. Check available disk space — the full mixture needs ~80-100 GB:
   ```bash
   python3 -c "import shutil; t,u,f=shutil.disk_usage('.'); print(f'Free: {f/1e9:.1f} GB  Used: {u/1e9:.1f} GB  Total: {t/1e9:.1f} GB')"
   ```
   If free space < 100 GB, warn the user before proceeding.

2. Verify HuggingFace login (required for gated DINOv3 weights, and used by `datasets`):
   ```bash
   huggingface-cli whoami
   ```

3. Download COCO images, LVIS zips (fixed URLs), and an OpenImages-V7 subset via FiftyOne:
   ```bash
   python data/download_and_prepare.py
   ```
   This can take a long time on first run. Monitor for FiftyOne dataset creation messages.

4. Convert and merge into a unified COCO-format label space:
   ```bash
   python data/convert_and_merge.py
   ```
   Outputs: `datasets/mixture/annotations/coco_merged/instances_{train,val}_merged.json`

5. Confirm final directory structure:
   ```bash
   python3 -c "
   import os
   paths = [
     'datasets/mixture/images/train2017',
     'datasets/mixture/images/val2017',
     'datasets/mixture/annotations/coco_merged/instances_train_merged.json',
     'datasets/mixture/annotations/coco_merged/instances_val_merged.json',
   ]
   for p in paths:
       status = 'OK' if os.path.exists(p) else 'MISSING'
       print(f'  [{status}] {p}')
   "
   ```
