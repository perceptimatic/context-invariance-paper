import argparse

from convolution_submission_gen_constants import *


# CMDLINE INTERFACE DEFINITION
def add_parser_single_job_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "original_submission_path",
        type=str,
        help=(
            "Path to the directory for the original submission from which we"
            " will create the convolution submission. This dir should be in"
            " standard zerospeech 2021 submission format, i.e. contain subdirs"
            " such as phonetic/dev-clean and phonetic/test-other."
        ),
    )

    parser.add_argument(
        "output_path",
        type=str,
        help=(
            "Path where the convolution submission will be written."
            " The script creates subdirs such as"
            " output_path/running_mean/av_ws_5/[submissionname]."
        ),
    )

    parser.add_argument(
        "--convolution_type",
        type=str,
        default=RUNNING_MEAN,
        choices=[RUNNING_MEAN, LAPLACIAN, BLUR_THEN_SHARPEN],
        help=(
            "Type of the convolution. running_mean blurs,"
            " laplacian sharpens edges."
        ),
    )

    parser.add_argument(
        "--window_s_running_mean",
        type=int,
        default=3,
        help=(
            "Window size for the running mean convolution."
            " (Laplacian only supports 3 and is assumed by default.)"
        ),
    )

    parser.add_argument(
        "--max_sharpen",  # Only applied on Laplacian
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    # Convenience option to copy the original meta.yaml to
    # the convolutional submission. Modify description as appropriate
    parser.add_argument(
        "--copy_meta",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
