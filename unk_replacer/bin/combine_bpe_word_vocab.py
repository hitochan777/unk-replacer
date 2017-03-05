import argparse
import json
from os import path


def define_parser(parser):
    parser.add_argument('--bpe-voc', required=True, type=str, help='Path to BPE vocab file in json format')
    parser.add_argument('--word-voc', required=True, type=str, help='Path to word vocab file in json format')
    parser.add_argument('--output', required=True, type=str, help="Path to output vocab file")
    parser.add_argument('--separator', default="@@", type=str, help="Separator for BPE")


def run(options):
    if path.isfile(options.output):
        input("%s will be overwritten. Press Enter to proceed" % (options.output, ))

    bpe_src_voc = []
    bpe_tgt_voc = []
    with open(options.bpe_voc) as bpe_voc:
        bpe_voc = json.load(bpe_voc)
        for s in bpe_voc[0]:
            if s[-1].endswith("</w>"):
                bpe_src_voc.append("".join(s))
            else:
                bpe_src_voc.append("".join(s) + options.separator)

        for s in bpe_voc[1]:
            if s[-1].endswith("</w>"):
                bpe_tgt_voc.append("".join(s))
            else:
                bpe_tgt_voc.append("".join(s) + options.separator)

    with open(options.word_voc) as f:
        word_voc = json.load(f)

    with open(options.output, 'w') as f:
        final_vocab = [bpe_src_voc + word_voc[0], bpe_tgt_voc + word_voc[1]]
        json.dump(final_vocab, f)


def command_line(args=None):
    parser = argparse.ArgumentParser(description='Combine word and BPE vocab', formatter_class=argparse.RawTextHelpFormatter)
    define_parser(parser)
    options = parser.parse_args(args)
    run(options)


if __name__ == "__main__":
    command_line()
