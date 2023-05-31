import argparse, sys
from typing import Any

## Verifies that the files listed in an item file for a dataset (such as test-clean) 
## and files listed in an alignment list for transcriptions match. 

## Created for the 2021 zerospeech dataset. 
## If you are using this in a later year, be careful to check that the 
## item file or the alignment file structures have not changed, and modify
## the script accordingly.

def file_list_d_from_item_file(itemfile_path) -> dict[str, Any]:
    with open(itemfile_path, 'r') as f:
        lines = f.readlines()
    d: dict[str, Any] = {}
    for l in lines[1:]:
        f_name = l.split(' ')[0]
        d.setdefault(f_name)
    return d

def file_list_d_from_file_list_aligned(file_path) -> dict[str, Any]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    d: dict[str, Any] = {}
    for l in lines:
        d.setdefault(l.strip('\n'))
    return d

def verify_match(item_file_path: str, file_list_path: str) -> bool:
    item_d = file_list_d_from_item_file(item_file_path)
    file_list_d = file_list_d_from_file_list_aligned(file_list_path)
    return item_d.keys() == file_list_d.keys()

def add_parser_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "item_file_path", type=str, help="Path to the item file."
    )
    parser.add_argument(
        "file_list_path", type=str, help="Path to the aligned file list."
    )

def main(argv):
    description = "Verify file list match between an item file and an aligned file list."
    parser = argparse.ArgumentParser(description=description)
    add_parser_args(parser)
    args = parser.parse_args(argv)

    print(f'File lists match: {verify_match(args.item_file_path, args.file_list_path)}')

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)


    