# zs2021-transcription-submission-gen

Generates a submission for zerospeech 2021 from one-hot encoded transcriptions.

There is an additional script, verify_all_subsets.py to verify that the transcription file list matches the files in the item files.

There is also the option to create submissions from the transcription with some deliberate errors added in. It will have a random chance that the boundary between two phonemes, such as G G G OW OW gets shifted to the right or left by some i. (E.g. one option is to generate a submission where there is a 50% chance that the boundary will move to the right and we will end up with G G G G OW) See gen_error_submissions.py.