import argparse
""" System Modules """

# ---------------- CLI ----------------
def build_myargparser():
    """
    Build the argument parser for the CLI.

    :return:
    """
    p = argparse.ArgumentParser(description="RAPTOR is ready to hunt. Start inference, training or preprocessing by passing a valid JSON congiuration file")
    p.add_argument('--config-file', type=str, required=True, help='Name to the JSON configuration file.')
    p.add_argument('--test-image', type=str, required=False, help='Complete path to the test image.')
    p.add_argument('--test-tags', type=str, required=False, help='Comma-separated list of tags for zero-shot inference.')


    return p
