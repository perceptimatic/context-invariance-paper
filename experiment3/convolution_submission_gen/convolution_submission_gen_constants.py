# CONSTANTS
SUBSETS = ["dev-clean", "dev-other", "test-clean", "test-other"]
PHONETIC = "phonetic"
SUBMISSION_D = "submission"
REPR_FILE_EXTENSION = ".txt"  # only format supported at this point
INPUT_DIMS = 2
LAPLACIAN_3 = (-1, 4, -1)
LAPLACIAN_3_MAX_SHARPEN = (-1, 8, -1)
LAPLACIAN_WS = 3

META_F_NAME = "meta.yaml"

# Convolution types
RUNNING_MEAN = "running_mean"
LAPLACIAN = "laplacian"
BLUR_THEN_SHARPEN = "blur_then_sharpen"

# WINDOW_SIZE_DIRNAMES
AV_WS_PREFIX = "av_ws_"
LAPL_WS_PREFIX = "lapl_ws_"
MAX_SHARPEN_SUFFIX = "_max_sharpen"
