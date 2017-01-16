import mojimoji
import argparse


class NumberHandler:

    @classmethod
    def process_number(cls, string: str, split_prefix='_'):
        if mojimoji.zen_to_han(string).replace(',', '').isdigit():
            integer = int(mojimoji.zen_to_han(string).replace(',', ''))
            if 0 <= integer <= 12:
                return "<@num:%d>" % integer
            elif 13 <= integer <= 99:
                return "<@num:2d>"
            elif 100 <= integer <= 999:
                return "<@num:3d>"
            elif 1000 <= integer <= 9999:
                return "<@num:4d>"
            else:
                assert integer >= 10000
                return "<@num:big>"

        split_words = cls.split_number(string, split_prefix=split_prefix)
        return ' '.join(split_words)

    @classmethod
    def split_number(cls, token, split_prefix='_'):
        assert isinstance(token, str)
        split = []
        for c in token:
            if len(split) == 0:
                split.append(c)
            else:
                if split[-1][-1].isnumeric() or c.isnumeric():
                    split.append(split_prefix + c)
                else:
                    split[-1] = split[-1] + c

        return split

    @classmethod
    def restore(cls, allow_invalid_seq=True, split_prefix='_'):
        assert isinstance(tokens, list)
        restored = []
        for token in tokens:
            if token.startswith(split_prefix):
                if len(restored) == 0:
                    if allow_invalid_seq:
                        restored.append("")
                    else:
                        raise RuntimeError("First token starts from %s" % split_prefix)

                restored[-1] = restored[-1] + token[len(split_prefix):]
            else:
                restored.append(token)

        return restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process tokens including numbers')
    parser.add_argument("input", type=str, help="Path to the file you want to process")
    parser.add_argument("action", choices=["split", "restore"], help='Type of number handler')

    args = parser.parse_args()

    splitter = NumberHandler
    with open(args.input, "r") as f:
        for line in f:
            tokens = line.rstrip().split(" ")
            if args.action == "split":
                split = []
                for token in tokens:
                    split.append(splitter.process_number(token))

                print(" ".join(split))
            elif args.action == "restore":
                restored = splitter.restore(tokens)
                print(" ".join(restored))
            else:
                raise NotImplementedError()
