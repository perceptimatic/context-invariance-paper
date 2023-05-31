# context-invariance-paper
Codebase for https://arxiv.org/pdf/2210.15775.pdf, forthcoming in Interspeech.

# Experiment 1
To run experiment 1 for a given model, go to the [Zero Resource Challenge Benchmark Toolkit](https://github.com/zerospeech/benchmarks) and follow the instructions for running the abx-LS benchmark. Alternatively, if you want more granular control, you can go directly to https://github.com/zerospeech/libri-light-abx2/. See also 

# Experiment 2
For a given model, run https://github.com/zerospeech/libri-light-abx2/ with the option `--pooling hamming` and compare with the default abx score (i.e. `--pooling none`). The code to generate 1-hot encoded submission from a transcription is demonstrated in the current repository, as is the generation of submissions from the 1-hot encoded submission where the phoneme boundaries are occasionally shifted.

# Experiment 3
The code for running experiment 3 is included in the current repository. To enable the suitable environment, do: 

```
cd experiment3
conda env create -f environment.yml
conda activate compute-map
```
