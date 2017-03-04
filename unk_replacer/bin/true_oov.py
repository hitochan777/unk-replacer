import argparse


def count_oov(train_path, test_path):
    nb_oov = 0
    with open(train_path) as train, open(test_path) as test:
        voc = set()
        oov = set()
        for line in train:
            tokens = line.strip().split(" ")
            for token in tokens:
                voc.add(token)

        nb_words_in_test = 0
        for line in test:
            tokens = line.strip().split(" ")
            nb_words_in_test += len(tokens)
            for token in tokens:
                if token not in voc:
                    nb_oov += 1
                    oov.add(token)

        print("%d/%d(%f%%) are OOV" % (nb_oov, nb_words_in_test, nb_oov/nb_words_in_test*100))
        print(oov)

def main(args=None):
    parser = argparse.ArgumentParser(description='Count OOV')
    parser.add_argument('train', type=str)
    parser.add_argument('test', type=str)

    options = parser.parse_args(args)
    count_oov(options.train, options.test)


if __name__ == "__main__":
    main()
