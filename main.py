import argparse
from contextlib import redirect_stdout


def main():
    pass


parser = argparse.ArgumentParser(description='Test Some Linear Classifier for BLG527E (ML) Class of ITU \'21 ')
parser.add_argument("-f", "--file-output", action="store_true",
                    help="Write output to a file instead of terminal.")
args = parser.parse_args()

if __name__ == '__main__':
    if args.file_output:
        with open('output', 'a+') as f:
            with redirect_stdout(f):
                main()
                print('\n')
    else:
        main()
