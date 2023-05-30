from enum import Enum, auto
from typing import NamedTuple

import numpy as np


## ConvolutionSubmissionGenerator models
class GeneratorArgs(NamedTuple):
    original_submission_path: str
    output_path: str
    convolution_type: str
    window_s_running_mean: int
    max_sharpen: bool
    copy_meta: bool


class Convolution(Enum):
    RUNNING_MEAN = auto()
    LAPLACIAN = auto()
    BLUR_THEN_SHARPEN = auto()  # i.e. running_mean then laplacian


class SubmissionFilePath(NamedTuple):
    file_name: str  # with extension
    absolute_path: str


class FileOutput(NamedTuple):
    absolute_file_path: str
    representations: np.ndarray


class SubsetOutput(NamedTuple):
    file_outputs: list[FileOutput]
