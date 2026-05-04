Run production inference using the unified closed + open-vocabulary detector.

Arguments: $ARGUMENTS
Usage: /infer <image_path> [--prompts "label1, label2, label3"]

Steps:
1. Parse $ARGUMENTS to extract the image path and optional --prompts string.
   If no image path is provided, ask the user for one.

2. Verify the checkpoint exists at `runs/dinov3_rtdetr/final/`:
   ```bash
   python3 -c "import os; print('OK' if os.path.exists('runs/dinov3_rtdetr/final') else 'ERROR: run /train first')"
   ```

3. Build and run the inference command:
   ```bash
   python serve/infer_cli.py \
     --image <image_path> \
     --lexicon resources/open_vocab_lexicon.txt \
     [--prompts "<labels>"]
   ```
   Omit `--prompts` if none were supplied (lexicon-only open-vocab mode).

4. Pretty-print the JSON result showing:
   - `labels`: unique detected label names
   - `detections`: list of {label, bbox [x,y,w,h], score, source (closed|open)}

Key thresholds (defined in `serve/predictor.py`):
- `CLOSED_SCORE_THRESH = 0.30`  — RT-DETR confidence gate
- `OPEN_SCORE_THRESH = 0.28`    — CLIP cosine similarity gate
- `TOPK_CLIP_LEXICON = 200`     — max lexicon candidates per image
