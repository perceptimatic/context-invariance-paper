import argparse
from io import TextIOWrapper
from pathlib import Path
import sys, os, yaml
import tqdm
from typing import NamedTuple
import json


PHONETIC_SUB_DIRS = ['dev-clean', 'dev-other', 'test-clean', 'test-other']
GOLD_LOC_DIC = "gold_loc_d.json"

class YamlResponse(NamedTuple):
    yaml_file: TextIOWrapper
    submission_data_path: str

class SubmissionParams(NamedTuple):
    metric: str
    frame_shift: float

class ItemFileLineData(NamedTuple):
    file_name: str
    start: float
    end: float
    gold_label: str

class FileOutput(NamedTuple):
    absolute_file_path: str
    representations: list[str]

class ExtractorArgs(NamedTuple):
    submission_path: str
    output_path: str
    item_file_path: str

def metayaml_file(main_submission_dir: str):
    f_name = 'meta.yaml'
    alts = [os.path.join(main_submission_dir, f_name)]
    # Some ad-hocery to deal with legacy and odd submission names
    alts.append(os.path.join(main_submission_dir, 'submission', f_name))
    alts.append(os.path.join(main_submission_dir, 'best_norm', f_name))
    alts.append(os.path.join(main_submission_dir, 'evaluation_3rd_best', f_name))
    for alt_loc in alts:
        if os.path.isfile(alt_loc):
            try:
                yaml_file = open(alt_loc, 'r')
            except:
                continue
            else:
                return YamlResponse(yaml_file, alt_loc.removesuffix(f_name))
    return None

def get_item_file_lines(item_file_path: str):
    if not os.path.isfile(item_file_path):
        return None
    try: 
        with open(item_file_path, 'r') as f:
            lines = f.readlines()
    except:
        return None
    else: 
        return lines
    
def process_item_file_line(ifld: ItemFileLineData,
                           args: ExtractorArgs,
                           phonetic_data_path: str,
                           file_loc_dict: dict[str, str],
                           frame_step: float):
    # Step 1: Find if a corresponding file exists in the submissions
    # NB! Not all lines are in the submission sets!
    if not ifld.file_name in file_loc_dict:
        return # continue
    # Step 2: and where
    subdir = file_loc_dict[ifld.file_name]
    data_file_path = os.path.join(
        phonetic_data_path,
        subdir,
        ifld.file_name + '.txt'
        )
    # Step 3: Open this file in the submission, read the lines
    if not os.path.isfile(data_file_path):
        raise IOError(f'No file at {data_file_path}, check submission integrity.')
    try:
        with open(data_file_path, 'r') as f:
            lines = f.readlines()
    except:
        raise IOError(f'Failed to read contents of file at {data_file_path}')
    # Step 4: calculate how many frames to skip at beginning
    f_start = n_frames_to_skip(ifld.start, frame_step) + 1
    # Step 5: calculate how many frames to include. 
    # And process_file / extract features (line=frame rep)
    reps = lines[f_start:f_start + n_frames_to_include(ifld.start, ifld.end, frame_step)]
    # Step6: save result to file in our output dir
    submissiondirname = os.path.basename(os.path.normpath(args.submission_path))
    output_f_name = f'{ifld.file_name}_{ifld.start}_{ifld.end}_{ifld.gold_label}'.replace('.', 'dot').replace(',', 'comma')
    out_path = os.path.join(args.output_path, submissiondirname, 'phonetic', subdir, output_f_name + '.txt')
    save_file_representations(FileOutput(out_path, reps))

def save_file_representations(file_output: FileOutput):
        d = os.path.dirname(file_output.absolute_file_path)
        Path(d).mkdir(parents=True, exist_ok=True)
        with open(file_output.absolute_file_path, 'a') as f:
            for frame in file_output.representations:
                f.write(frame)

# Not too slow, just load it every time for now.
# Plus, it provides a way to check submission integrity
def file_loc_dict(phonetic_data_path: str):
    # We'll create a dictionary, {filename: subdir} where subdir € PHONETIC_SUB_DIRS
    loc_d: dict[str, str] = {}
    for subdir in PHONETIC_SUB_DIRS:
        subdir_path = os.path.join(phonetic_data_path, subdir)
        if not os.path.isdir(subdir_path):
            raise IOError(f'Unable to find {subdir} in submission.')
        # get only the text files
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            if not os.path.isfile(file_path):
                raise IOError(f"Found a dir at {file_path}, should only include files?")
            if not file_name.lower().endswith('.txt'):
                raise NotImplementedError(f"Unexpected extension for {file_path}.")
            loc_d[file_name.strip('.txt')] = subdir
    return loc_d

def n_frames_to_skip(seconds_to_skip: float, frame_step: float):
    return int(seconds_to_skip / frame_step) # frame_step aka ~ feature size

def n_frames_to_include(start: float, end: float, frame_step: float):
    duration = end - start # in s
    return int(duration / frame_step)

def get_submission_params(yaml_file: TextIOWrapper, submissiondirname: str):
    try:
        with yaml_file:
            meta = yaml.safe_load(yaml_file.read())
            metric = meta['parameters']['phonetic']['metric']
            frame_shift = float(meta['parameters']['phonetic']['frame_shift'])
    except:
        raise IOError(f"Data retrieval from yaml file failed for {submissiondirname}.")
    else:
        return SubmissionParams(metric, frame_shift)


def process_submission(args: ExtractorArgs,
                       gold_location_d: dict[str, str]):
    submissiondirname = os.path.basename(os.path.normpath(args.submission_path))
    yaml_response = metayaml_file(args.submission_path)
    if not yaml_response:
        raise FileNotFoundError(f"No yam_file for {submissiondirname}")
    yaml_file = yaml_response.yaml_file
    submission_params = get_submission_params(yaml_file, submissiondirname)
    phonetic_data_path = os.path.join(yaml_response.submission_data_path, 'phonetic')
    if not os.path.isdir(phonetic_data_path):
        raise IOError(f"Not a dir: {phonetic_data_path}.")

    item_file_lines = get_item_file_lines(args.item_file_path)
    if not item_file_lines:
        raise ValueError(f"Failed to retrieve item file lines at {args.item_file_path}.")
    loc_d = file_loc_dict(phonetic_data_path)
    if not loc_d:
        raise ValueError("Failed to construct a location dictionary for the submission files.")
    if not loc_d == gold_location_d:
        print("WARNING! Check submission integrity, location dictionary does not match gold. Continuing extraction ...")
    for line in tqdm.tqdm(item_file_lines, mininterval=60, maxinterval=70):
        # This is also where we are going get the name for our feature file: '{file_name}_{start}_{end}_{gold_label}...'
        file_name, start, end, gold_label = line.split()
        item_file_line_data = ItemFileLineData(file_name, float(start), float(end), gold_label)
        process_item_file_line(item_file_line_data, args, phonetic_data_path, loc_d, submission_params.frame_shift)
    print('... DONE.')

def add_parser_single_job_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "submission_path",
        type=str,
        help=("Path to the directory for the submission for which we will extract the features for the map computation.")
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="Path where the features for the map calculation will be written. A dir with the original submission name will be created here."
    )

    parser.add_argument(
        "item_file_path",
        type=str, 
        help="Path to the item file containing word level splits, no hapax." # words_split_nohapax
    )

def main(argv):
    description = ("Extract features from a submission and save them word by word. This is used for the map calculation.")
    parser = argparse.ArgumentParser(description=description)
    add_parser_single_job_args(parser)
    cmdlineargs = parser.parse_args(argv)
    args = ExtractorArgs(cmdlineargs.submission_path,
                         cmdlineargs.output_path,
                         cmdlineargs.item_file_path)
    print(f'... Feature extraction ... Args:\n {args}')
    with open(GOLD_LOC_DIC, "r") as f:
        gold_loc_d = json.load(f)
    process_submission(args, gold_loc_d)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
