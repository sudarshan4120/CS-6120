from collections import Counter, defaultdict
import numpy as np
import math
import random

"""
CS6120 Homework 2 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
    # STUDENTS IMPLEMENT
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams
    pass


def read_file(path: str) -> list:
    """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
    # PROVIDED
    f = open(path, "r", encoding="utf-8")
    contents = f.readlines()
    f.close()
    return contents


def tokenize_line(line: str, ngram: int,
                  by_char: bool = True,
                  sentence_begin: str = SENTENCE_BEGIN,
                  sentence_end: str = SENTENCE_END):
    """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
    # PROVIDED
    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
        # otherwise split on white space
        inner_pieces = line.split()

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens


def tokenize(data: list, ngram: int,
             by_char: bool = True,
             sentence_begin: str = SENTENCE_BEGIN,
             sentence_end: str = SENTENCE_END):
    """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
    # PROVIDED
    total = []
    # also glue on sentence begin and end items
    for line in data:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
        total += tokens
    return total


class LanguageModel:

    def __init__(self, n_gram):
        """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
        # STUDENTS IMPLEMENT
        self.n = n_gram
        self.model = {}
        self.n_gram_counts = defaultdict(Counter)
        self.total_counts = Counter()
        self.vocabulary = set()

        pass

    def train(self, tokens: list, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
        # STUDENTS IMPLEMENT
        token_counts = Counter(tokens)
        tokens = [token if token_counts[token] > 1 else UNK for token in tokens]

        # Update vocabulary
        self.vocabulary.update(tokens)

        # Generate n-grams and update counts
        n_grams = create_ngrams(tokens, self.n)
        for n_gram in n_grams:
            if len(n_gram) < self.n:  # Skip incomplete n-grams
                continue
            prefix, word = n_gram[:-1], n_gram[-1]
            self.n_gram_counts[prefix][word] += 1
            self.total_counts[prefix] += 1
            print(self.total_counts)
            print(self.n_gram_counts)

        if verbose:
            print(f"Model trained with {self.n}-gram")
        pass

    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model

    Returns:
      float: the probability value of the given tokens for this model
    """
        # STUDENTS IMPLEMENT
        probability = 1.0
        V = len(self.vocabulary)  # Vocabulary size for Laplace smoothing

        # Generate n-grams from the sentence tokens
        n_grams = create_ngrams(sentence_tokens, self.n)

        for n_gram in n_grams:
            if len(n_gram) < self.n:  # Skip incomplete n-grams
                continue
            prefix, word = n_gram[:-1], n_gram[-1]
            # Laplace smoothing applied
            word_given_prefix_count = self.n_gram_counts[prefix][word] + 1  # Add-one smoothing
            prefix_count = self.total_counts[prefix] + V  # Add V for Laplace smoothing
            n_gram_probability = word_given_prefix_count / prefix_count
            probability *= n_gram_probability  # Multiply the probabilities of each n-gram

        return probability
        pass

    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
      #STUDENTS IMPLEMENT
        sentence = []
        # Initialize the context with the start tokens
        context = [SENTENCE_BEGIN] * (self.n - 1)

        while True:
          # Get the possible next words and their probabilities for the current context
          next_words = self.model.get(context, {})
          if not next_words:
            break  # In case of no available next words, stop the generation

          # Convert the next words and their counts into probabilities
          total_count = sum(next_words.values())
          probabilities = [count / total_count for count in next_words.values()]

          # Randomly choose the next word based on the probabilities
          next_word = random.choices(list(next_words.keys()), weights=probabilities, k=1)[0]

          if next_word == SENTENCE_END:
            break  # End of sentence
          sentence.append(next_word)

          # Update the context for the next iteration
          context = context[1:] + [next_word]  # Slide the context window

        return sentence


    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
        # PROVIDED
        return [self.generate_sentence() for i in range(n)]

    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
    Returns:
      float: the perplexity value of the given sequence for this model
    """

        pass


# not required
if __name__ == '__main__':
    print()
    # print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")
    # print("call a function")
    # print(tokenize_line("tokenize this sentence!", 2, by_char=False))
    # print(tokenize(["apples are fruit", "bananas are too"], 2, by_char=False))

    n = 1
    lm = LanguageModel(n)
    sentences = read_file("training_files/unknowns_mixed.txt")
    tokens = tokenize(sentences, n, by_char=False)
    lm.train(tokens)
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    print(lm.n)

    print(lm.score("<s> flamingo".split()))
