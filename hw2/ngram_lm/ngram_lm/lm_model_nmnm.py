from collections import Counter
import numpy as np
import os

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
    n_grams = []
    for i in range(0, len(tokens) - n + 1):
        one_group = tokens[i:i + n]
        n_grams.append(tuple(one_group))

    return n_grams


def read_file(path: str) -> list:
    """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """

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
    _training_file = fr'{os.getcwd()}/training_files/berp-training.txt'

    def __init__(self, n_gram: int, training_file: str = None):
        """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
      training_file (str): (optional) path to training file
                            DEFAULT set to ./training_files/berp-training.txt
    """
        # Attribute declaration
        self.unk_filtered = None
        self.n_gram = n_gram
        self.model = {}
        self.vocab = None
        self.vocabulary_size = 0

        if training_file:
            LanguageModel.set_filename(training_file)

        # Reading the file
        train_str_lst = read_file(LanguageModel.get_filename())
        self.tokens = tokenize(train_str_lst, n_gram, by_char=False)

    def train(self, tokens: list = None, verbose: bool = False, unk_replace: bool = True) -> None:
        """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    The model is a dictionary of dictionary. The outer dictionary captures context and
    the inner one captures word occurring after the given context. The value of the inner
    dictionary keys are counts of that word occurring in the given context from outer dictionary

    Args:
        tokens: (list): (optional) tokens
        unk_replace: (bool): default value True, replace 1 frequency tokens with UNK during training
        verbose (bool): default value False, to be used to turn on/off debugging prints
    """
        if tokens:
            self.tokens = tokens

        self.vocab = Counter(self.tokens)

        # UNK modification logic
        if unk_replace:
            self.unk_filtered = dict(filter(lambda val: val[1] <= 1, self.vocab.items()))
            self.vocab = dict(filter(lambda val: val[1] > 1, self.vocab.items()))

            if self.unk_filtered:
                self.vocab[UNK] = sum(self.unk_filtered.values())

                if verbose:
                    print("Modified Vocabulary with ", UNK)
                    print(f"{self.vocab[UNK]=}")

                # Replacing tokens itself with UNK
                self.tokens = list(map(lambda x: UNK if x in self.unk_filtered.keys() else x, self.tokens))
                if verbose:
                    print("Replaced tokens with ", UNK)

        self.vocabulary_size = len(set((self.vocab.keys())))

        grams = create_ngrams(self.tokens, self.n_gram)
        if verbose:
            debug_gram_cnt = 0
            print(f"Created {len(grams)} ngrams")

        print_history = []

        # Training the model with ngram window
        for window in grams:
            context = window[:-1]
            word = window[-1]

            if context not in self.model.keys():
                self.model[context] = {}
            if word not in self.model[context]:
                self.model[context][word] = 0

            self.model[context][word] += 1

            if verbose:
                percent_complete = round((debug_gram_cnt / len(grams)) * 100)

                if percent_complete % 20 == 0 and percent_complete not in print_history:
                    print(f"Training progress {percent_complete}%")
                    print_history.append(percent_complete)
                debug_gram_cnt += 1

    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a sentence token to be scored by this model

    Returns:
      float: the probability value of the given tokens for this model
    """
        # replace unknown tokens with UNK
        sentence_tokens = [UNK if token in self.unk_filtered.keys() else token for token in sentence_tokens]
        sentence_tokens = [UNK if token not in self.vocab.keys() else token for token in sentence_tokens]
        sentence = ([SENTENCE_BEGIN] * (self.n_gram - 1)) if self.n_gram > 1 else [SENTENCE_BEGIN]
        print((sentence))
        grams = create_ngrams(sentence_tokens, self.n_gram)
        prob = 1

        for token_set in grams:
            context = token_set[:-1]
            word = token_set[-1]

            # Some logic for UNK context or words
            try:
                if context:
                    c_sentence = self.model[context][word]
            except KeyError:
                c_sentence = 0  # If this is an unknown word in the given context

            try:
                if not context:
                    c_sentence = self.vocab[word]
            except KeyError:
                c_sentence = self.vocab[UNK]

            try:
                c_context = sum(self.model[context].values())
            except KeyError:
                c_context = 0

            # Laplace Smoothing
            set_prob = (c_sentence + 1) / (c_context + self.vocabulary_size)

            prob *= set_prob


        return prob

    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.

    Returns:
      list: the generated sentence as a list of tokens
    """
        print('hellllllllo')
        sentence = ([SENTENCE_BEGIN] * (self.n_gram - 1)) if self.n_gram>1 else [SENTENCE_BEGIN]
        print((sentence))
        word_chosen = None
        print('!!!!!!!!!!!!!!!!!!!')
        # Generate words until we get sentence end
        while word_chosen != SENTENCE_END:
            # Finding list of all possible words for given context
            print('!!!!!!!!!!!!!!!!!!!')
            window_start = len(sentence) - self.n_gram + 1

            gram_window = tuple(sentence[window_start:])
            print(f'{gram_window=}')
            words_possible = self.model.get(gram_window, {})
            print(f'{words_possible=}')

            # Filter out SENTENCE_BEGINS
            words_possible = dict(filter(lambda item: item[0] != SENTENCE_BEGIN, words_possible.items()))
            print(f'{words_possible=}')
            total_context_words = sum(words_possible.values())
            print(f'{total_context_words=}')

            # Probability calculation
            probablities = [(v / total_context_words) for v in words_possible.values()]
            print(f'{probablities=}')
            words_possible = list(words_possible.keys())
            print(f'{words_possible=}')

            # Choosing the word using Shannon technique
            words_possible_idx = list(range(len(words_possible)))
            print(f'{words_possible_idx=}')
            word_chosen_idx = np.random.choice(words_possible_idx, size=1, p=probablities)
            print(f'{words_possible_idx=}')

            word_chosen = words_possible[word_chosen_idx[0]]
            print(f'{word_chosen=}')
            sentence.append(word_chosen)
            print('sent',sentence)


        sentence = sentence + ([SENTENCE_END]*(self.n_gram-2))
        print('while loop ends')
        print(f'{sentence=}')

        # sentence = sentence[self.n_gram - 1:-1]
        return sentence

    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
        print('aaaaaaaaaaa')
        return [self.generate_sentence() for _ in range(n)]

    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
    Returns:
      float: the perplexity value of the given sequence for this model
    """
        one_by_score = 1 / self.score(sequence)
        perplexity = one_by_score ** (1 / len(sequence))

        return perplexity

    @classmethod
    def get_filename(cls):
        return cls._training_file

    @classmethod
    def set_filename(cls, file):
        cls._training_file = file
        return 1


if __name__ == '__main__':
    # Script testing
    print("tokenize_line", tokenize_line("tokenize this sentence!", 3, by_char=False))
    print("tokenize", tokenize(["apples are fruit", "bananas are too"], 2, by_char=False))
    print("create_ngrams", create_ngrams(['<s>', 'apples', 'are', 'bananas', 'too', '</s>'], 4), '\n\n')

    '''ng = LanguageModel(3)
    ng.train(verbose=True)
    print(ng.score("<s>".split()))
    print(*ng.generate(5), sep="\n")
    print(ng.perplexity(ng.generate_sentence()))'''

    # n = 1
    # lm = LanguageModel(n)
    # sentences = read_file("training_files/iamsam.txt")
    # tokens = tokenize(sentences, n, by_char=False)
    # lm.train(tokens)
    # # ((2 + 1) / (10 + 5))
    # print(lm.score("<s>".split()))
    # print(lm.model)

    n = 2
    lm = LanguageModel(n)
    sentences = read_file("training_files/iamsam.txt")
    tokens = tokenize(sentences, n, by_char=False)
    lm.train(tokens)
    # ((2 + 1) / (10 + 5))**2
    lm.generate(3)