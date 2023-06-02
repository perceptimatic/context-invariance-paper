# context-invariance-paper
Codebase for https://arxiv.org/pdf/2210.15775.pdf, forthcoming in Interspeech.

# Install
1) Clone this repository.
2) 
```
cd context-invariance-paper
conda env create -f environment.yml
conda activate abx-exp23
```

This installs what you need to run experiments 2-3. As outlined below, **Experiment 1** is run separately.

# Experiment 1
To run experiment 1 for a given model, go to the [Zero Resource Challenge Benchmark Toolkit](https://github.com/zerospeech/benchmarks) and follow the instructions for running the abx-LS benchmark. Alternatively, if you want more granular control, you can go directly to https://github.com/zerospeech/libri-light-abx2/.

# Experiment 2
1) GENERATING ABX SUBMISSIONS FROM THE TRANSCRIPTION. This repository includes the code to generate a 1-hot-encoded abx submission from the transcription. You can also generate several 1-hot encoded submissions from the transcription such that some errors are deliberately added in, specifically with the phoneme boundaries occasionally shifted. To generate these submissions, do the following:

```
conda activate abx-exp23
python experiment2/gen_transcription_submission.py [output_path]
python experiment2/gen_error_submissions.py [output_path]
```

2) RUNNING THE COMPARISON. Once you have the 1-hot-encoded submissions, or if you want to test another submission, run https://github.com/zerospeech/libri-light-abx2/ with the option `--pooling hamming` and then compare with the default abx score (i.e. `--pooling none`). For these submissions, use one of the clean subsets and set the following options:

```
--feature_size 0.01
--speaker_mode within
--context_mode all
```

This will compute the score for both the within-context and without-context conditions.

# Experiment 3
1) GENERATING SUBMISSIONS WITH A BLURRING FILTER APPLIED. To generate a modified submission from a given submission, do
```
conda activate abx-exp23
python experiment3/convolution_submission_gen/convolution_submission_gen.py -h
```
and follow the instructions. You will want to run with `--convolution_type running_mean` and `--window_s_running_mean 3` (and {5,7}).

2) RUNNING THE EVALUATION ON THE SUBMISSIONS. Once you have the submissions from above, run
```
python experiment3/mapcode/compute_map_from_dir.py
```
