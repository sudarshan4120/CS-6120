# import random
# from collections import Counter
# import numpy as np
# import math
#
# """
# CS6120 Homework 2 - starter code
# """
#
# # constants
# SENTENCE_BEGIN = "<s>"
# SENTENCE_END = "</s>"
# UNK = "<UNK>"
#
#
# # UTILITY FUNCTIONS
#
# def create_ngrams(tokens: list, n: int) -> list:  # correct=================================
#     """Creates n-grams for the given token sequence.
#   Args:
#     tokens (list): a list of tokens as strings
#     n (int): the length of n-grams to create
#
#   Returns:
#     list: list of tuples of strings, each tuple being one of the individual n-grams
#   """
#     # STUDENTS IMPLEMENT
#     ngrams = []
#     for i in range(len(tokens) - n + 1):
#         ngram = tuple(tokens[j] for j in range(i, i + n))
#         ngrams.append(ngram)
#     return ngrams
#     pass
#
#
# def read_file(path: str) -> list:
#     """
#   Reads the contents of a file in line by line.
#   Args:
#     path (str): the location of the file to read
#
#   Returns:
#     list: list of strings, the contents of the file
#   """
#     # PROVIDED
#     f = open(path, "r", encoding="utf-8")
#     contents = f.readlines()
#     f.close()
#     return contents
#
#
# def tokenize_line(line: str, ngram: int,
#                   by_char: bool = True,
#                   sentence_begin: str = SENTENCE_BEGIN,
#                   sentence_end: str = SENTENCE_END):
#     """
#   Tokenize a single string. Glue on the appropriate number of
#   sentence begin tokens and sentence end tokens (ngram - 1), except
#   for the case when ngram == 1, when there will be one sentence begin
#   and one sentence end token.
#   Args:
#     line (str): text to tokenize
#     ngram (int): ngram preparation number
#     by_char (bool): default value True, if True, tokenize by character, if
#       False, tokenize by whitespace
#     sentence_begin (str): sentence begin token value
#     sentence_end (str): sentence end token value
#
#   Returns:
#     list of strings - a single line tokenized
#   """
#     # PROVIDED
#     inner_pieces = None
#     if by_char:
#         inner_pieces = list(line)
#     else:
#         # otherwise split on white space
#         inner_pieces = line.split()
#
#     if ngram == 1:
#         tokens = [sentence_begin] + inner_pieces + [sentence_end]
#     else:
#         tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
#     # always count the unigrams
#     return tokens
#
#
# def tokenize(data: list, ngram: int,
#              by_char: bool = True,
#              sentence_begin: str = SENTENCE_BEGIN,
#              sentence_end: str = SENTENCE_END):
#     """
#   Tokenize each line in a list of strings. Glue on the appropriate number of
#   sentence begin tokens and sentence end tokens (ngram - 1), except
#   for the case when ngram == 1, when there will be one sentence begin
#   and one sentence end token.
#   Args:
#     data (list): list of strings to tokenize
#     ngram (int): ngram preparation number
#     by_char (bool): default value True, if True, tokenize by character, if
#       False, tokenize by whitespace
#     sentence_begin (str): sentence begin token value
#     sentence_end (str): sentence end token value
#
#   Returns:
#     list of strings - all lines tokenized as one large list
#   """
#     # PROVIDED
#     total = []
#     # also glue on sentence begin and end items
#     for line in data:
#         line = line.strip()
#         # skip empty lines
#         if len(line) == 0:
#             continue
#         tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
#         total += tokens
#     return total
# class LanguageModel:
#
#     def __init__(self, n_gram):
#         """Initializes an untrained LanguageModel
#     Args:
#       n_gram (int): the n-gram order of the language model to create
#     """
#         # STUDENTS IMPLEMENT
#
#         self.n_gram = n_gram
#         self.n_gram_counts = {}
#         self.context_counts = {}  # model
#         self.token_counts = {}  # Stores counts of individual tokens
#
#     # ****************************************************************************************************************
#     def train(self, tokens: list, verbose: bool = False) -> None:
#         """Trains the language model on the given data. Assumes that the given data
#             has tokens that are white-space separated, has one sentence per line, and
#             that the sentences begin with <s> and end with </s>
#         Args:
#         tokens (list): tokenized data to be trained on as a single list
#         verbose (bool): default value False, to be used to turn on/off debugging prints"""
#
#         # Replace single-occurrence tokens with UNK in the token list
#         self.token_counts = Counter(tokens)
#         print(f"{self.token_counts=}")
#         tokens = [UNK if self.token_counts[t] < 2 else t for t in tokens]
#         self.token_counts = Counter(tokens)
#         # print(tokens)
#
#         # Create and count n-grams with UNK replacements
#         n_grams = create_ngrams(tokens, self.n_gram)
#         # print(n_grams)
#         for n in n_grams:
#             self.n_gram_counts[n] = self.n_gram_counts.get(n, 0) + 1
#
#         # print(self.n_gram_counts)
#
#         if self.n_gram > 1:
#             context = create_ngrams(tokens, self.n_gram - 1)
#             for n in context:
#                 self.context_counts[n] = self.context_counts.get(n, 0) + 1
#
#         # print(self.context_counts)
#         print(f"{self.n_gram_counts}")
#         print(self.context_counts)
#
#         # Verbose output
#         if verbose:
#             print(f"{self.n_gram=}")
#             print(f"{self.n_gram_counts=}")
#             print(f"{self.token_counts=}")
#             print(f"{self.context_counts=}")
#             print(f"Trained on {len(tokens)} tokens, with {len(tokens)} after UNK replacement.")
#             print(f"Total unique {self.n_gram}-grams: {len(self.n_gram_counts)}")
#             if self.n_gram > 1:
#                 print(f"Total unique {self.n_gram - 1}-grams: {len(self.context_counts)}")
#
#
#
# lm = LanguageModel(2)
#
# sentences = read_file("training_files/iamsam.txt")
# tokens = tokenize(sentences, 2, by_char=False)
# lm.train(tokens, verbose=True)
#
sentence_tokens = ['this', 'is', 'an', 'example', 'sentence']
token_counts = {'this': 1, 'is': 2, 'an': 1, 'sentence': 1}
UNK = 'UNKNOWN'
for i in range(len(sentence_tokens)):
    if sentence_tokens[i] not in token_counts:
        sentence_tokens[i] = UNK  # Assigning UNK if the token is not in token_counts


print(sentence_tokens)



