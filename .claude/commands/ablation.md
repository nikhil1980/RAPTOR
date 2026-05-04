Run the focal loss (α, γ) ablation grid search and report LVIS slice metrics.

Arguments: $ARGUMENTS

Steps:
1. Warn the user: this trains one short run per (α, γ) grid point (default 3 epochs each). Estimated GPU time is O(grid_size × 3 epochs). Confirm they want to proceed.

2. Check GPU and WANDB_PROJECT:
   ```bash
   python3 -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.device_count(), 'devices')"
   echo "WANDB_PROJECT=${WANDB_PROJECT:-<not set — set before running>}"
   ```

3. Run the ablation sweep (pass any extra flags from $ARGUMENTS):
   ```bash
   python train/run_ablation.py --config-file config.json $ARGUMENTS
   ```

4. After completion, display the results table:
   | α    | γ   | AP   | APr (rare) | APc (common) | APf (frequent) |
   |------|-----|------|-----------|-------------|----------------|
   | ...  | ... | ...  | ...        | ...          | ...            |

5. Highlight the (α, γ) setting with the best APr (rare class performance is typically the optimization target).
   Recommend updating `config.json` `loss.focal_alpha` and `loss.focal_gamma` with the best values.
