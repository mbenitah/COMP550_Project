import argparse
import numpy as np
import random
import os
import sys
import string
import re
import pickle
import masking


from time import time
from datetime import datetime

NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

# #############################################################################
#
# Parser
#
# #############################################################################


def get_arguments():
    def _str_to_bool(s):
        """Convert string to boolean (in argparse context)"""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _percentage(p):
        try:
            if p[-1] == '%':
                p = float(p[:-1]) / 100
            else:
                p = float(p)

            if not 0 <= p <= 1:
                raise ValueError()

            return p

        except ValueError:
            print('Please enter a value between 0 and 1, got {}.'.format(p))

    parser = argparse.ArgumentParser(description='Predicting masked text.')
    # required parameters
    parser.add_argument('text_file', type=str,
                        help='Full path or relative path to text file')

    # optional parameters
    parser.add_argument('--percent_words', type=_percentage, default=0.01,
                        help='Percentage of words to mask in the input text file.')

    # flags
    parser.add_argument('-w', action='store_true', default=False,
                        help='Use fine tuned weights for prediction. '
                        'If False, model will use pretrained weights.')
    return parser.parse_args()


def read(filename):
    '''Returns a string read from a text file.
    Input:
    filename:   full or relative path of the text file
    Output:
    txt:        string with text from filename'''

    with open (filename, "r") as f:
        txt = f.read()
    return txt

def main():
    args = get_arguments()
    pretrained = False

    print('=' * 80)
    print('File: {}'.format(args.text_file))
    print('Masking {:.2%} of the words'.format(args.percent_words))
    if not args.w:
        print('Using pretrained weights')
        pretrained = True
    else:
        print('Using fine-tuned model')
    print('-' * 80)

    txt = read(args.text_file)
    tokenized_text, masked, ids = masking.mask(txt, args.percent_words)
    masking.predict(tokenized_text, masked, pretrained=pretrained)


if __name__ == '__main__':
    main()
