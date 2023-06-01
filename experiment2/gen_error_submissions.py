import argparse, sys, os
import numpy as np
import gen_transcription_submission as gen

SEED = 3451
ERROR_CHANCE = 0.5

class ErrorSubmissionGenerator(gen.TranscriptionSubmissionGenerator):
    def __init__(self,
                 boundary_shift: int):
        self.boundary_shift = boundary_shift

    ## Subclass method implementations
    def with_errors(self,
                    phoneme_tokens: list[str]) -> list[str]:
        ban_i = -100
        for i, t in enumerate(phoneme_tokens):
            # We want to avoid recursive application of boundary shifts.
            # Otherwise, if we shifted the boundary to the right by say 1 
            # in one round of the loop, we would find in the next round 
            # a new boundary, and so on, and if we keep hitting error 
            # success events we could garble a long section of phonemes.
            if i <= ban_i:
                continue

            l = len(phoneme_tokens)
            # index safety
            if i + 1 + self.boundary_shift >= l or i + 1 >= l:
                continue
            next_t = phoneme_tokens[i+1] 
            transition = t != next_t
            if not transition:
                continue
            mess_it_up = np.random.binomial(1, ERROR_CHANCE)
            if not mess_it_up:
                continue
            # We either shift right or left. Changing phoneme_tokens[i] 
            # counts as shifting the boundary left since the transition is 
            # between phoneme_tokens[i] and phoneme_tokens[i + 1]
            go_right = self.boundary_shift > 0
            invader_t = t if go_right else next_t
            start = 1 if go_right else self.boundary_shift + 1 
            stop = self.boundary_shift + 1 if go_right else 1
            # 1 is excluded, so range(-3,1) will give us -3,-2,-1,0 for the shift
            for shift_i in range(start, stop):
                phoneme_tokens[i + shift_i] = invader_t
            if go_right:
                ban_i = i + self.boundary_shift
        return phoneme_tokens
    
def add_parser_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "output_path", 
        type=str, 
        help="Path where the submission(s) will be written."
    )
    
def main(argv):
    np.random.seed(SEED)
    description = ("Generate a zerospeech phonetic submission " 
        "– for all subsets (test-clean etc.) – from the " 
        "transcriptions with some errors deliberately added in.")
    parser = argparse.ArgumentParser(description=description)
    add_parser_args(parser)
    cmdlineargs = parser.parse_args(argv)
    transcriptions_top_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'transcriptions'))

    for i in (-10, -8, -6):
        if i == 0:
            continue
        sg = ErrorSubmissionGenerator(i)
        output_path = f'{cmdlineargs.output_path}-randomshift{i}'
        args = gen.GeneratorArgs(transcriptions_top_path,
                                 output_path,
                                 gen.TRANSCRIPTION_SUBMISSION_MAP,
                                 gen.TRANSCRIPTION_TEXTDIRNAME,
                                 gen.TRANSCRIPTION_FILENAME,
                                 gen.FILE_LIST_ALIGNMENT_FILENAME)
        sg.generate_submission(args)
        print(f"""
              Generating 1-hot encoded transcription submission with some shifted boundaries. 
              Params:\n{args}\n, i={i}\n
              """)
    print("\nDONE. Submissions with boundary shifts generated.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)