import json
import argparse


def print_unk_stat(fn, voc, max_nb_ex=None, msg=None):
    assert isinstance(voc, set)
    nb_unk = 0
    nb_total = 0
    with open(fn) as lines:
        for index, line in enumerate(lines):
            if max_nb_ex is not None and index >= max_nb_ex:
                break

            tokens = line.rstrip("\n").split(' ')
            nb_total += len(tokens)
            for token in tokens:
                if token not in voc:
                    nb_unk += 1

    if msg is not None:
        print(msg)

    print("%d/%d(%f%%) is unknown" % (nb_unk, nb_total, float(nb_unk)/nb_total*100))


def main(args=None):
    parser = argparse.ArgumentParser(description='Get statistics of unknown words')
    parser.add_argument('config', type=str, help='Path to config file made by make_data in knmt')

    options = parser.parse_args(args)

    with open(options.config, 'r') as config_fs:
        config = json.load(config_fs)

    voc_fn = config['save_prefix'] + '.voc'
    with open(voc_fn, 'r') as voc_fs:
        voc = json.load(voc_fs)
        src_voc = set(voc[0]['voc_lst'])
        tgt_voc = set(voc[1]['voc_lst'])

    train_src_fn = config['src_fn']
    train_tgt_fn = config['tgt_fn']
    dev_src_fn = config['dev_src']
    dev_tgt_fn = config['dev_tgt']
    test_src_fn = config['test_src']
    test_tgt_fn = config['test_tgt']

    max_nb_ex = config['max_nb_ex']

    if train_src_fn is not None:
        print_unk_stat(train_src_fn, voc=src_voc, max_nb_ex=max_nb_ex, msg='train src')

    if train_tgt_fn is not None:
        print_unk_stat(train_tgt_fn, voc=tgt_voc, max_nb_ex=max_nb_ex, msg='train tgt')

    if dev_src_fn is not None:
        print_unk_stat(dev_src_fn, voc=src_voc, max_nb_ex=max_nb_ex, msg='dev src')

    if dev_tgt_fn is not None:
        print_unk_stat(dev_tgt_fn, voc=tgt_voc, max_nb_ex=max_nb_ex, msg='dev tgt')

    if test_src_fn is not None:
        print_unk_stat(test_src_fn, voc=src_voc, max_nb_ex=max_nb_ex, msg='test src')

    if test_tgt_fn is not None:
        print_unk_stat(test_tgt_fn, voc=tgt_voc, max_nb_ex=max_nb_ex, msg='test tgt')

if __name__ == "__main__":
    main()
