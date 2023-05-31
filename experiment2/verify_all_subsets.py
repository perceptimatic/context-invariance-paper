import verify_file_list_match as vm
import sys, argparse, os

## For all subsets, verifies that item files match with the alignment files
## of the transcripts.

def add_parser_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "item_files_path", type=str, help="Path to the item files."
    )
    parser.add_argument(
        "file_lists_path", type=str, help="Path to the aligned file lists."
    )

def main(argv):
    description = "Verify file list match between an item file and an aligned file list, for all subsets."
    parser = argparse.ArgumentParser(description=description)
    add_parser_args(parser)
    args = parser.parse_args(argv)

    data_subdirs = ['valid-clean', 'valid-other', 'test-clean', 'test-other']
    item_subdirs = ['dev-clean', 'dev-other', 'test-clean', 'test-other']
    matches: list[bool] = []
    print('...')
    for i, datadirname in enumerate(data_subdirs):
        file_list_path = os.path.join(args.file_lists_path, datadirname, 'file_list_aligned.txt')
        itemdirname = item_subdirs[i]
        item_path = os.path.join(args.item_files_path, itemdirname, f'{itemdirname}.item')
        matches.append(vm.verify_match(item_path, file_list_path))
    print(f'Data dir: {args.file_lists_path}')
    print(f'Subdirs: {data_subdirs}')
    print(f'Item files dir: {args.item_files_path}')
    print(f'Subdirs: {item_subdirs}')
    print(f'Matches between file lists in data dir and file lists in item files dir:\n {matches}')
        
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)