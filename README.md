# Replacer

Unknown word replacer in Neural Machine Translation (NMT)

[![wercker status](https://app.wercker.com/status/f85bf4841c6422cd5ddfba7bdf635318/s/ "wercker status")](https://app.wercker.com/project/byKey/f85bf4841c6422cd5ddfba7bdf635318)

## Requirements
- Python >=3.5

## Install

```
make install
make test (Optional)
```

## What you need in the experiment

- training data
- (optional) development data
- test data

## Basic Usage

1. Build source and target vocabulary from the training data with the following command

```bash
    python build_vocab.py :TODO
```

2. Get word alignment for the parallel corpora and lexical translation tables for both direction.
   
   Typically you can obtain the lexical translation tables as byproducts 
   of word alignment. 
   You can use GIZA++ or mgiza because they are fast.
   However, we recommend that you use [Nile](https://TODO.com),
   which is a supervised alignment model rather than GIZA++, 
   because it produces much better alignment.

2. Train source and target Word2vec models with the following command
 
```bash
    python train_word2vec.py :TODO
```

3. Replace unknown words in the training and development data with the
    following command.
   
```bash
    python train_replacer.py TODO:
```

4. Train NMT model with the replaced training data from step 3

```bash
    TODO
```

5. Replace unknown words in the test data with the following command.

```bash
    python test_replacer.py TODO:
```

6. Translate the replaced test data with the trained NMT model.

We recommend that you ensemble several models because it normally
leads to the better attention.

7. Restore the final translation with the following command.

```bash
    python restorer.py
```

## End-to-End Experiment with Sacred

TBW
