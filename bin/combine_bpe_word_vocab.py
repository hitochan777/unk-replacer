import argparse
import json
from os import path


def main(args=None):
    parser = argparse.ArgumentParser(description='Combine BPE and word vocab')
    parser.add_argument('--bpe-src-voc', required=True, type=str, help='Path to source BPE vocab file')
    parser.add_argument('--bpe-tgt-voc', required=True, type=str, help='Path to target BPE vocab file')
    parser.add_argument('--word-voc', required=True, type=str, help='Path to word vocab file in json format')
    parser.add_argument('--output', required=True, type=str, help="Path to output vocab file")
    parser.add_argument('--separator', default="@@", type=str, help="Separator for BPE")

    options = parser.parse_args(args)

    if path.isfile(options.output):
        input("%s will be overwritten. Press Enter to proceed" % (options.output, ))

    bpe_src_voc = []
    bpe_tgt_voc = []
    with open(options.bpe_src_voc) as lines:
        for line in lines:
            rule = line.strip(" \t\n").split(" ")
            if rule[-1].endswith("</w>"):
                bpe_src_voc.append("".join(rule))
            else:
                bpe_src_voc.append("".join(rule) + options.separator)

    with open(options.bpe_tgt_voc) as lines:
        for line in lines:
            rule = line.strip(" \t\n").split(" ")
            if rule[-1].endswith("</w>"):
                bpe_tgt_voc.append("".join(rule))
            else:
                bpe_tgt_voc.append("".join(rule) + options.separator)

    with open(options.word_voc) as f:
        word_voc = json.load(f)

    with open(options.output, 'w') as f:
        final_vocab = [bpe_src_voc + word_voc[0], bpe_tgt_voc + word_voc[1]]
        json.dump(final_vocab, f)

if __name__ == "__main__":
    main()
