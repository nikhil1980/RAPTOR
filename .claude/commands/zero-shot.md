Run zero-shot prompted open-vocabulary inference on a single image.

Arguments: $ARGUMENTS
Usage: /zero-shot <image_path> "<tag1,tag2,tag3>"

Steps:
1. Parse $ARGUMENTS: first positional arg is the image path, second is a comma-separated tag string.
   If either is missing, ask the user for the missing value.

2. Run zero-shot inference:
   ```bash
   python eval/zero_shot_prompt.py \
     --config-file config.json \
     --test-image <image_path> \
     --test-tags "<tag1,tag2,tag3>"
   ```

3. Display detected objects, their bounding boxes, and confidence scores.

Notes:
- Tags are matched against detected region crops using SigLIP/CLIP cosine similarity.
- The OPEN_SCORE_THRESH (0.28 cosine sim) gates what gets returned.
- If no checkpoint exists yet, tell the user to run `/train` first.
