Run RAPTOR evaluation on the validation set and report COCO + LVIS metrics.

Arguments: $ARGUMENTS

Steps:
1. Verify the checkpoint exists:
   ```bash
   python3 -c "import os; d='runs/dinov3_rtdetr/final'; print('Checkpoint OK' if os.path.exists(d) else 'ERROR: No checkpoint at '+d)"
   ```
   If missing, tell the user to run `/train` first.

2. Run evaluation (pass any extra flags from $ARGUMENTS):
   ```bash
   python eval/eval_and_infer.py --config-file config.json $ARGUMENTS
   ```

3. After eval completes, surface the key metrics:
   - Look for COCO mAP lines in stdout (format: `AP: X.XX`)
   - Look for LVIS slice lines: `APr`, `APc`, `APf`
   - Check for `coco_detections_val.json` and any `*.png` PR curve plots saved in the working directory

4. Summarise in a table:
   | Metric | Value |
   |--------|-------|
   | COCO mAP@[0.50:0.95] | ... |
   | LVIS APr (rare) | ... |
   | LVIS APc (common) | ... |
   | LVIS APf (frequent) | ... |
