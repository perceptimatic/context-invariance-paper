import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from convolution_submission_gen_cmdline import *
from convolution_submission_gen_constants import *
from convolution_submission_gen_errors import *
from convolution_submission_gen_model import *

# This script generates a convolution submission, i.e. new representations
# from existing representations, with a convolutional filter applied.

# TYPE ALIASES
ConvolutionFunc = Callable[[int, np.ndarray, GeneratorArgs], np.ndarray]

# SUBMISSION GENERATOR
class ConvolutionSubmissionGenerator:
    def generate_submission(self, args: GeneratorArgs):
        print(f"Generating submission. Args: {args}")
        print("...")
        for s in tuple(
            os.path.join(args.original_submission_path, PHONETIC, sub)
            for sub in SUBSETS
        ):
            if not os.path.isdir(s):
                # try 'submission' subdir
                subdirname = os.path.basename(os.path.normpath(s))
                s = os.path.join(
                    args.original_submission_path,
                    SUBMISSION_D,
                    PHONETIC,
                    subdirname,
                )
                if not os.path.isdir(s):
                    raise FileNotFoundError(SUBDIR_NOT_FOUND_ERROR)
            self._save_subset_representations(self._subset_output(s, args))
            print(f"Subset {s} DONE.\n...")
        if args.copy_meta:
            self._copy_meta(args)
        print("Submission generated.")

    # Private methods
    def _subset_output(
        self, subset_path: str, args: GeneratorArgs
    ) -> SubsetOutput:
        file_outputs: list[FileOutput] = []
        for f in self._find_all_subset_files(subset_path, REPR_FILE_EXTENSION):
            file_outputs.append(self._file_output(f, args))
        return SubsetOutput(file_outputs)

    def _file_output(
        self, f: SubmissionFilePath, args: GeneratorArgs
    ) -> FileOutput:
        m: np.ndarray = np.loadtxt(f.absolute_path)
        assert len(m.shape) == INPUT_DIMS
        f_out_path = self.file_out_path(f, args)
        convolution_func = self._get_convolution_func(args.convolution_type)
        out_m = self._convolved(m, args, convolution_func)
        return FileOutput(f_out_path, out_m)

    def _get_convolution_func(self, convolution_type: str) -> ConvolutionFunc:
        match self._convolution_type(convolution_type):
            case Convolution.RUNNING_MEAN:
                return self._running_mean
            case Convolution.LAPLACIAN:
                return self._laplacian
            case Convolution.BLUR_THEN_SHARPEN:
                return self._blur_then_sharpen
            case other:
                raise ValueError("Unsupported convolution type.")

    def _convolved(
        self, m: np.ndarray, args: GeneratorArgs, convolution_f: ConvolutionFunc
    ):
        """Runs a convolution (running mean or laplacean) of step 1 and a given
        window size over all the frames.
        Padding added at the beginning and end during the convolution operation.
        m: 2D numpy array where m[i] is the vector representation of frame i.
        Returns: a numpy array of the same dimensions as m (padding removed).
        """
        if convolution_f == self._blur_then_sharpen:
            running_m = self._convolved(m, args, self._running_mean)
            return self._convolved(running_m, args, self._laplacian)

        padded_m = self._padded(m, self._padding_size(args, convolution_f))
        conv_m = np.zeros(m.shape)
        for i in range(conv_m.shape[0]):
            # window centered on row i of the non-padded array
            conv_m[i] = convolution_f(i, padded_m, args)
        return conv_m

    # Convolution functions
    def _running_mean(
        self, i: int, padded_m: np.ndarray, args: GeneratorArgs
    ) -> np.ndarray:
        return padded_m[i : i + args.window_s_running_mean].mean(0)

    def _laplacian(
        self, i: int, padded_m: np.ndarray, args: GeneratorArgs
    ) -> np.ndarray:
        lapl = LAPLACIAN_3_MAX_SHARPEN if args.max_sharpen else LAPLACIAN_3
        # ONLY window size 3 supported for now
        return np.array(lapl).dot(padded_m[i : i + 3])

    def _blur_then_sharpen(
        self, i: int, padded_m: np.ndarray, args: GeneratorArgs
    ) -> np.ndarray:
        raise ValueError(UNREACHABLE_ERROR)

    def _padding_size(self, args: GeneratorArgs, convolution_f: ConvolutionFunc):
        window_s = None
        if convolution_f == self._running_mean:
            self._check_window_odd(args.window_s_running_mean)
            window_s = args.window_s_running_mean
        elif convolution_f == self._laplacian:
            window_s = 3  # The only supported value for now
        elif convolution_f == self._blur_then_sharpen:
            self._check_window_odd(args.window_s_running_mean)
            # we run the mean/blurring first
            window_s = args.window_s_running_mean
        else:
            raise NotImplementedError
        return int(np.floor(window_s / 2))

    def _padded(self, m: np.ndarray, padding_n: int) -> np.ndarray:
        padded_m = m.copy()
        for _ in range(padding_n):
            padded_m = np.insert(padded_m, 0, np.zeros(m.shape[1]), axis=0)
            padded_m = np.append(padded_m, [np.zeros(m.shape[1])], axis=0)
        return padded_m

    def _check_window_odd(self, window_s: int):
        if window_s % 2 == 0:
            raise NotImplementedError(WINDOW_SIZE_EVEN_ERROR)

    def _convolution_type(self, convolution: str) -> Convolution:
        if convolution == RUNNING_MEAN:
            return Convolution.RUNNING_MEAN
        elif convolution == LAPLACIAN:
            return Convolution.LAPLACIAN
        elif convolution == BLUR_THEN_SHARPEN:
            return Convolution.BLUR_THEN_SHARPEN
        else:
            raise ValueError(CONVOLUTION_TYPE_ERROR)

    # IO
    def _copy_meta(self, args: GeneratorArgs):
        yaml_in_path = self.metayaml_file_path(args.original_submission_path)
        if not yaml_in_path:
            print(META_YAML_WARNING)
            return
        submissiondirname = os.path.basename(
            os.path.normpath(args.original_submission_path)
        )
        yaml_out_path = os.path.join(
            args.output_path,
            args.convolution_type,
            self.window_s_name(args),
            submissiondirname,
            META_F_NAME,
        )
        shutil.copy(yaml_in_path, yaml_out_path)

    def metayaml_file_path(self, main_submission_dir: str) -> str | None:
        alts = [os.path.join(main_submission_dir, META_F_NAME)]
        # try 'submission' subdir
        alts.append(os.path.join(main_submission_dir, SUBMISSION_D, META_F_NAME))
        for alt_loc in alts:
            if os.path.isfile(alt_loc):
                return alt_loc
        return None

    def _save_subset_representations(self, subset_output: SubsetOutput):
        for f in subset_output.file_outputs:
            self._save_file_representations(f)

    def _save_file_representations(self, file_output: FileOutput):
        d = os.path.dirname(file_output.absolute_file_path)
        Path(d).mkdir(parents=True, exist_ok=True)
        with open(file_output.absolute_file_path, "a") as f:
            for frame in file_output.representations:
                out_s = " ".join(str(dim) for dim in frame)
                f.write(f"{out_s}\n")

    def _find_all_subset_files(
        self, path_dir: str, extension: str
    ) -> list[SubmissionFilePath]:
        out: list[SubmissionFilePath] = []
        for root, dirs, filenames in os.walk(path_dir):
            for f in filenames:
                if f.endswith(extension):
                    out.append(SubmissionFilePath(f, os.path.join(root, f)))
        return out

    def file_out_path(self, f: SubmissionFilePath, args: GeneratorArgs) -> str:
        submissiondirname = os.path.basename(
            os.path.normpath(args.original_submission_path)
        )
        subset_dir = os.path.basename(os.path.dirname(f.absolute_path))
        conv_type = args.convolution_type
        return os.path.join(
            args.output_path,
            conv_type,
            self.window_s_name(args),
            submissiondirname,
            PHONETIC,
            subset_dir,
            f.file_name,
        )

    def window_s_name(self, args: GeneratorArgs) -> str:
        window_s = ""
        match self._convolution_type(args.convolution_type):
            case Convolution.RUNNING_MEAN:
                window_s = f"{AV_WS_PREFIX}{args.window_s_running_mean}"
            case Convolution.LAPLACIAN:
                # The ONLY supported window size for now
                window_s = f"{LAPL_WS_PREFIX}{LAPLACIAN_WS}"
                if args.max_sharpen:
                    window_s += MAX_SHARPEN_SUFFIX
            case Convolution.BLUR_THEN_SHARPEN:
                window_s = (
                    f"{AV_WS_PREFIX}{args.window_s_running_mean}"
                    f"_{LAPL_WS_PREFIX}{LAPLACIAN_WS}"
                )
                if args.max_sharpen:
                    window_s += MAX_SHARPEN_SUFFIX
            case other:
                raise ValueError(CONVOLUTION_TYPE_ERROR)
        return window_s


def main(argv: list[str]):
    description = (
        "Generate a modified zerospeech phonetic submission from an existing"
        " submission by running a convolution (e.g. blurring or sharpening) of"
        " a given window size, step 1, over the submission."
    )
    parser = argparse.ArgumentParser(description=description)
    add_parser_single_job_args(parser)
    cmdlineargs = parser.parse_args(argv)
    args = GeneratorArgs(
        cmdlineargs.original_submission_path,
        cmdlineargs.output_path,
        cmdlineargs.convolution_type,
        cmdlineargs.window_s_running_mean,
        cmdlineargs.max_sharpen,
        cmdlineargs.copy_meta,
    )
    sg = ConvolutionSubmissionGenerator()
    sg.generate_submission(args)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
