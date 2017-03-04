#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yusuke Oda


import sys
import argparse


def create_parser():
  parser = argparse.ArgumentParser(description="restore BPE subwords")

  parser.add_argument(
    '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
    metavar='PATH',
    help="Input file (default: standard input).")
  parser.add_argument(
    '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
    metavar='PATH',
    help="Output file (default: standard output)")
  parser.add_argument(
      '--use-separator', '-e', action='store_true', default=False,
      help="Each subword is appended by separator (default: '%(default)s')"
  )
  parser.add_argument(
      '--separator', '-s', type=str, default='@@', metavar='STR',
      help="Separator between non-final subword units (default: '%(default)s')"
  )
  parser.add_argument(
      '--eow', '-w' , type=str, default='</w>', metavar='STR',
      help="End of word token (default: '%(default)s')"
  )

  return parser


def main():
  parser = create_parser()
  args = parse_args()


if __name__ == '__main__':
  parser = create_parser()
  args = parser.parse_args()

  for line in args.input:
    result = ''
    for word in line.strip().split(" "):
        if args.use_separator:
            if word.endswith(args.separator):
                word = word[:-len(args.separator)]
            else:
                word += ' '
        else:
            if word.endswith(args.eow):
                word = word[:-len(args.eow)] + ' '

        result += word

    args.output.write(result.strip())
    args.output.write('\n')
