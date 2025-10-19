import os
import json
import time
from pathlib import Path
from typing import List, Dict
from PIL import Image
import torch
""" System Modules """

from serve.predictor import Predictor
from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config
""" User Modules """

"""
Script to run zero-shot object detection using DINOv3 + RTDETR with OV head.

@author: Nikhil Bhargava
@date: 2025-10-19
@license: Apache-2.0
@description: This script performs zero-shot object detection using a pre-trained DINOv3 + RTDETR model with an open-vocabulary head. It encodes text prompts and matches them against detected objects in images.
@copyright: Copyright 2025 Nikhil Bhargava
"""

logger = get_logger(__name__)





def main():
    """
    Main function to run the workflow.

    :return:
    """
    parser = build_myargparser()
    args = parser.parse_args()

    # 1. Load configuration from JSON file if provided
    if args.config_file:
        # Get the parent path
        par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Construct the full path to the config file
        config_file_path = os.path.join(par_path, str(args.config_file))

        # Check if the file exists before attempting to open
        if os.path.exists(config_file_path):
            try:
                # 1. Load the environment variables
                load_env_from_json(config_file_path)
                Config.ROOT = os.getenv("RAPTOR_PATHS_ROOT")
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_PRODUCTION")}] "
                            f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

                logger.debug(f"---> Configuration loaded from: {args.config_file} <---")

                # Get the parent path
                par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

                # Get the trained model path
                output_dir = os.path.join(par_path, os.getenv("RAPTOR_PATHS_MODEL_DIR",
                                                              "runs/dinov3_rtdetr"))
                model_dir= os.path.join(output_dir, "final")

                # Lexicon path
                lexicon_path = os.path.join(par_path, os.getenv("RAPTOR_PATHS_LEXICON_PATH",
                                                                "resources/open_vocab_lexicon.txt"))
                # 2. Call Zero-shot Inference
                start_time = time.perf_counter()

                if args.test_image is None or not os.path.isfile(args.test_image) or args.test_tags is None:
                    logger.error(f"---> PLEASE PROVIDE BOTH --test-image AND --test-tags ARGUMENTS. "
                                 f"SWITCHING TO DEFAULT <---")
                    test_image = os.path.join(par_path, "datasets/mixture/images/val2017/000000039769.jpg")
                    test_tags = ["mic stand", "mixing console", "stage light", "ukulele"]
                else:
                    test_image = args.test_image
                    test_tags = args.test_tags.split(",")

                logger.info(f"---> RUNNING ZERO-SHOT INFERENCE ON IMAGE: {test_image} WITH TAGS: {test_tags} <---")

                pred = Predictor(model_dir=model_dir, lexicon_path=lexicon_path, use_openclip=True)
                result = pred.predict(args.image, general_prompts=test_tags, return_boxes=True)
                
                logger.info(f"---> ZERO-SHOT DETECTION RESULTS: {json.dumps(result, indent=4)} <---")

                end_time = time.perf_counter()
                logger.debug(f"---> DINOv3 WITH RTDETR OV HEAD ZERO-SHOT ON "
                             f"IMAGE: {test_image} WITH TAGS: {test_tags} IN "
                             f"TIME: {(end_time - start_time): .3f} SECONDS <---")

            except json.JSONDecodeError:
                logger.exception(f"---> ERROR: INVALID JSON in {args.config_file} <---")

            except Exception as e:
                logger.exception(f"---> ERROR READING CONFIGURATION: {e} <---")
        else:
            logger.info(f"---> CONFIGURATION FILE: '{args.config_file}' NOT FOUND AT PATH: {config_file_path}")
    else:
        logger.info(f"---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")

    logger.info(f"---> FINISHED [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_PRODUCTION")}] "
                f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

if __name__ == '__main__':
    main()
