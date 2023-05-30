# This code is a slightly modified version of code by Robin Algayres

import tqdm
import argparse
import sys
import os
from pathlib import Path
import torch
from torch import nn
import numpy as np
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MapResults:
    source: str
    MAP_meanpooling: float
    MAP_maxpooling: float


def normalize(data):
    '''Normalize numpy array data in-place.'''
    epsilon = 0.00000001
    norm = np.sqrt(np.sum(np.power(data, 2), axis=1))
    data /= (norm[:, None]+epsilon)
    return data


def map_at_r(embeddings, labels):
    r"""
    Compute the MAP@R as defined in section 3.2 of https://arxiv.org/pdf/2003.08505.pdf

    Parameters:
        - embeddings (2D numpy array): shape = (N,d), contains the embeddings to evaluate
        - labels (1D numpy array): shape =(N,), contains the labels.
                                   Each element should be an integer.
    Returns:
        - mean_average_precision_at_r (float): the value of the MAP@R
    """

    k_faiss = np.bincount(labels.astype(int)).max()
    k_faiss = int(k_faiss)
    if k_faiss > 2047:  # if k_faiss is larger than 2048
        # it runs on CPU
        k_faiss = 2047

    # Initialize the calculator
    calculator = AccuracyCalculator(
        include=(), exclude=(), avg_of_avgs=False, k=k_faiss)
    # Insure that the type of the numpy array is float32 for faiss
    X = np.float32(embeddings)
    X = normalize(X)  # embeddings mut be normalize before FAISS
    # because FAISS only compute the dot
    # product and not the cosine distance
    if np.isnan(X).any() or np.isinf(X).any():
        print(np.sum(X))
        print('error')
        sys.exit()

    y = np.float32(labels)
    # Compute the MAP@R
    metric_name = 'mean_average_precision'
    score = calculator.get_accuracy(
        query=X,
        reference=X,
        query_labels=y,
        reference_labels=y,
        embeddings_come_from_same_source=True,
        include=[metric_name],
    )
    return score[metric_name]

def save_results(map_results: MapResults, output_path: str):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(output_path, 'map_results.json')
    with open(result_file, "w") as f:
        json.dump(vars(map_results), f, indent=2)

def add_parser_single_job_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "feature_dir",
        type=str,
        help=("Path to the directory where the features are organized by word.")
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="Path where the results of the map calculation will be written. A dir with the original submission name will be created here."
    )

def main(argv):
    #feature_dir=sys.argv[1] # path to source directory
    description = ("Compute and save map results to file.")
    parser = argparse.ArgumentParser(description=description)
    add_parser_single_job_args(parser)
    cmdlineargs = parser.parse_args(argv)
    # dev-clean only for now
    sub_dir = 'phonetic/dev-clean'
    feature_dir = os.path.join(cmdlineargs.feature_dir, sub_dir)
    assert os.path.isdir(feature_dir)
    featuredirname = os.path.basename(os.path.normpath(cmdlineargs.feature_dir))
    output_path = os.path.join(cmdlineargs.output_path, featuredirname, sub_dir)
    
    labels, label2ind = [], {}
    maxpool = []
    meanpool = []
    c = 0

    print("Date and time of run start:", datetime.now().strftime("%d/%m/%Y %H:%M"))
    print(f'Loading features from {feature_dir}')
    for fid in tqdm.tqdm(os.listdir(feature_dir), mininterval=30, maxinterval=39):
        if '.txt' not in fid:
            continue
        file_path = os.path.join(feature_dir, fid)

        frames = []
        with open(file_path) as buf:
            for line in buf:
                frames.append([float(frame)
                              for frame in line.rstrip().split(' ')])
            if len(frames) == 0:
              continue
        frames = torch.tensor(frames)
        maxpool.append(torch.max(frames, dim=0).values.unsqueeze(0))
        meanpool.append(torch.mean(frames, dim=0).unsqueeze(0))

        label = fid.split('_')[-1].split('.')[0]
        if label not in label2ind:
            label2ind[label] = len(label2ind)+1
        labels.append(label2ind[label])
        #c+=1
        #if c>100:
        #    break
    maxpool = torch.cat(maxpool, dim=0).numpy()
    meanpool = torch.cat(meanpool, dim=0).numpy()
    labels = torch.tensor(labels).numpy()
    print('computing map on', maxpool.shape)
    map_value_meanpool = map_at_r(meanpool, labels)
    print('MAP (meanpooling):', np.around(map_value_meanpool, 3))
    map_value_maxpool = map_at_r(maxpool, labels)
    print('MAP (maxpooling):', np.around(map_value_maxpool, 3))
    map_results = MapResults(feature_dir, map_value_meanpool, map_value_maxpool)
    save_results(map_results, output_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
