import json
import argparse
from collections import Counter
from operator import itemgetter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args=None):
    parser = argparse.ArgumentParser(description='Get sorted vocab')
    parser.add_argument("vocab", type=str, help="JSON format vocab file")
    parser.add_argument("source", type=str, help="Source side training file")
    parser.add_argument("target", type=str, help="Target side training file")
    parser.add_argument("output", type=str, help="Path to output")

    options = parser.parse_args(args)

    with open(options.vocab) as vocab_stream, open(options.source) as source_lines, open(options.target) as target_lines, open(options.output, "w") as output:

        logger.info("Loading vocab...")
        vocab = json.load(vocab_stream)
        src_voc = set(vocab[0])
        tgt_voc = set(vocab[1])

        logger.info("Counting source side tokens ...")
        src_voc_counter = Counter(token for line in source_lines for token in line.strip().split(" ") if token in src_voc)
        logger.info("Counting target side tokens ...")
        tgt_voc_counter = Counter(token for line in target_lines for token in line.strip().split(" ") if token in tgt_voc)
        
        logger.info("Getting the sorted vocab...")
        sorted_src_voc = list(map(lambda x: x[0], src_voc_counter.most_common()))
        sorted_tgt_voc = list(map(lambda x: x[0], tgt_voc_counter.most_common()))
    
        assert len(sorted_src_voc) == len(src_voc), (len(sorted_src_voc), len(src_voc))
        assert len(sorted_tgt_voc) == len(tgt_voc), (len(sorted_tgt_voc), len(tgt_voc))

        assert set(sorted_src_voc) == src_voc
        assert set(sorted_tgt_voc) == tgt_voc

        logger.info("Writing sorted vocab...")
        json.dump([sorted_src_voc, sorted_tgt_voc], output)


if __name__ == "__main__":
    main()
