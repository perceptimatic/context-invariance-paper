import argparse, os, sys
from pathlib import Path
from typing import NamedTuple
import one_hot_encoder as ohencoder

TRANSCRIPTION_SUBMISSION_MAP = {'valid-clean': 'dev-clean',
                                'valid-other': 'dev-other',
                                'test-clean': 'test-clean',
                                'test-other': 'test-other'}
TRANSCRIPTION_TEXTDIRNAME = 'text'
TRANSCRIPTION_FILENAME = 'text_phone_forced.txt'
FILE_LIST_ALIGNMENT_FILENAME = 'file_list_aligned.txt'

class GeneratorArgs(NamedTuple):
    transcriptions_top_path: str
    output_path: str
    transcription_submission_map: dict[str, str]
    transcription_textdirname: str
    transcription_filename: str
    filelist_filename: str

class TranscriptionPath(NamedTuple):
    transcription_path: str
    subset: str

class PhonemeTypesResponse(NamedTuple):
    phoneme_types: set[str]
    transcription_paths: list[TranscriptionPath]

class OutputAbsolutePaths(NamedTuple):
    output: str
    phonetic: str
    subpaths: tuple[str]

class TranscriptionSubmissionGenerator:
    
    def all_phoneme_types(self, args: GeneratorArgs) -> PhonemeTypesResponse:
        all_phoneme_types: set[str] = set()
        transcription_paths: list[TranscriptionPath] = []
        for subset in args.transcription_submission_map.keys():
            transcription_path = os.path.join(args.transcriptions_top_path, 
                                            subset, 
                                            args.transcription_textdirname, 
                                            args.transcription_filename)
            transcription_paths.append(TranscriptionPath(transcription_path, subset))
            with open(transcription_path, 'r') as f:
                lines = f.readlines()
            phoneme_tokens = []
            for l in lines:
                phoneme_tokens.extend(l.strip('\n').split(' '))
            all_phoneme_types = all_phoneme_types.union(set(phoneme_tokens))
        return PhonemeTypesResponse(all_phoneme_types, transcription_paths)
    
    def generate_submission(self, args: GeneratorArgs):
        phoneme_types, transcription_paths = self.all_phoneme_types(args)
        encoder_dict = ohencoder.encoder_dict(phoneme_types)
        self.create_submission_dirstructure(self.submission_absolute_paths(args))
        for transcription_path in transcription_paths:
            self.process_subset(args, transcription_path, encoder_dict)

    def process_subset(self,
                       args: GeneratorArgs,
                       transcription_path: TranscriptionPath,
                       encoder_dict: dict[str, ohencoder.OHEncoding]):
        with open(transcription_path.transcription_path, 'r') as f:
            # Each line corresponds to a file (the filenames can be found
            # in file_list_aligned.txt)
            transcription_lines = f.readlines()
        alignment_file_path = os.path.join(args.transcriptions_top_path,
                                           transcription_path.subset,
                                           args.filelist_filename)
        with open(alignment_file_path, 'r') as f:
            filelist = [l.strip('\n') for l in f.readlines()]
        for i, l in enumerate(transcription_lines):
            # One line in the transcription will correspond to one output file
            phoneme_tokens = l.strip('\n').split(' ')
            phoneme_tokens = self.with_errors(phoneme_tokens)
            ohencoded_tokens = [ohencoder.encode_phoneme(p,
                                                        encoder_dict
                                                        ) for p in phoneme_tokens]
            
            save_filename = '{}.txt'.format(filelist[i])
            self.save_representations(args,
                                      save_filename,
                                      transcription_path,
                                      ohencoded_tokens)

    def with_errors(self,
                    phoneme_tokens: list[str]) -> list[str]:
        """ Can be used in a subclass to deliberately add errors. 
        In the main class, simply return the tokens as is."""
        return phoneme_tokens

    def save_representations(self,
                             args: GeneratorArgs,
                             save_filename: str,
                             transcription_path: TranscriptionPath,
                             ohencoded_tokens: list[ohencoder.OHEncoding]):
        subset = args.transcription_submission_map[transcription_path.subset]
        output_file = os.path.join(args.output_path, 'phonetic', subset, save_filename)
        with open(output_file, 'a') as f:
            for phoneme in ohencoded_tokens:
                out_s = ' '.join(str(e) for e in phoneme)
                f.write(f'{out_s}\n')

    def create_submission_dirstructure(self, paths: OutputAbsolutePaths):
        l = [paths.output, paths.phonetic]
        l.extend(paths.subpaths)
        for p in l:
            Path(p).mkdir(parents=True, exist_ok=True)

    def submission_absolute_paths(self, args: GeneratorArgs):
        subsets = tuple(args.transcription_submission_map.values())
        phonetic = os.path.join(args.output_path, 'phonetic')
        subpaths = tuple(os.path.join(phonetic, s) for s in subsets)
        return OutputAbsolutePaths(args.output_path, phonetic, subpaths)
        
def add_parser_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--transcriptions_top_path", 
        type=str, 
        default='/scratch1/projects/zerospeech/2021/datasets/librispeech-preprocessed/',
        help=("Path to the main transcriptions directory that contains subdirs" 
        "such as test-clean etc.")
    )

    parser.add_argument(
        "output_path", 
        type=str, 
        help="Path where the submission will be written."
    )

def main(argv):
    description = ("Generate a zerospeech phonetic submission from the " 
    "transcriptions, for all subsets (test-clean etc.).")
    parser = argparse.ArgumentParser(description=description)
    add_parser_args(parser)
    cmdlineargs = parser.parse_args(argv)
    
    sg = TranscriptionSubmissionGenerator()
    args = GeneratorArgs(cmdlineargs.transcriptions_top_path,
                        cmdlineargs.output_path, 
                        TRANSCRIPTION_SUBMISSION_MAP,
                        TRANSCRIPTION_TEXTDIRNAME, 
                        TRANSCRIPTION_FILENAME,
                        FILE_LIST_ALIGNMENT_FILENAME)
    sg.generate_submission(args)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)