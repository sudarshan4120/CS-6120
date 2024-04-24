from collections import Counter
import random
import math
import numpy as np

"""

Homework 3 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS


# Used to create ngram tokens
def create_ngrams(tokens: list, n: int) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  # STUDENTS IMPLEMENT
  n_grams = []
  for i in range(len(tokens) - (n-1)):
    n_grams.append(tuple(tokens[i:i+n]))
  return n_grams

def read_file(path):
    """
    Reads the contents of a file line by line.
    Args:
        path (str): the location of the file to read
    Returns:
        list: list of strings, the contents of the file
    """
    contents = []
    with open(path, "r", encoding="latin-1") as f:
        contents = f.readlines()
    return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
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

def create_unk(tokens:list) -> list:
    count = Counter(tokens)
    unk_tokens = []
    for token in tokens:
        if count[token] <= 1:
            unk_tokens.append(UNK)
        else:
            unk_tokens.append(token)
    return unk_tokens

def tokenize(data: list, ngram: int, 
                   by_char: bool = False, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
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
    
    self.ngram = n_gram
    self.tokenCount = None
    self.ngramCount = None
    self.individualTokenCount = None
    self.ngramProbability = {}
  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    # STUDENTS IMPLEMENT
    
    # Tracks token count
    self.individualTokenCount = Counter(tokens)
    tokens = create_unk(tokens)
    
    # Unigram Case
    if(self.ngram <= 1):
        self.tokenCount = Counter(tokens)
        tokens = [token for token in tokens if token != '<s>']
        self.ngramCount = Counter(tokens)
        
        for unigram in dict(self.ngramCount):
            prob = (self.ngramCount[unigram] / len(tokens))
            self.ngramProbability[unigram] = prob
    
    # N-gram Case
    else:
        # Track the Ngram token and N-1gram token counts
        self.tokenCount = Counter(create_ngrams(tokens, self.ngram - 1))
        self.ngramCount = Counter(create_ngrams(tokens, self.ngram))
    
        for ngram in  dict(self.ngramCount):

            prob = (self.ngramCount[ngram]) / (self.tokenCount[ngram[:-1]])       
            self.ngramProbability[ngram] = prob
    
    


  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    # STUDENTS IMPLEMENT
    
    total_prob = 1
    vocabSize = len(self.tokenCount)
    for i in range(len(sentence_tokens)):
        if self.individualTokenCount[sentence_tokens[i]] <= 1:
            sentence_tokens[i] = UNK
    
    # Unigram Case
    if(self.ngram <= 1):
        for i in range(len(sentence_tokens)):
            total_prob = total_prob * ( (self.tokenCount[sentence_tokens[i]] + 1) / (sum(self.tokenCount.values()) + vocabSize))
    
    # Ngram Case
    else:
        for i in range(len(sentence_tokens) - (self.ngram - 1)):
            
            target = sentence_tokens[i:i+self.ngram]          
            total_prob = total_prob * ( (self.ngramCount[tuple(target)] + 1) / (self.tokenCount[tuple(target[:-1])] + vocabSize))
    
    
    return total_prob

  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    
    # Pad SENTENCE_BEGIN
    sentence = [SENTENCE_BEGIN] * max(self.ngram - 1, 1)
    nextWord = None
    
    # Generate until SENTENCE_END
    while nextWord != SENTENCE_END:
        
        # Unigram Case
        if(self.ngram <= 1):
            nextWord = random.choices(list(self.ngramProbability.keys()), weights = list(self.ngramProbability.values()), k = 1)[0]
            sentence.append(nextWord)
        
        # Only happens in only UNK vocab scenario:
        elif(len(self.ngramProbability.items()) == 1):
            if(len(sentence) > 5):
                nextWord = SENTENCE_END
                sentence.append(nextWord)
                
            else:
                nextWord = UNK
                sentence.append(nextWord)                  
        
        # Ngram Case
        else:
            nextWordDict = { item[0][-1]: item[1] for item in self.ngramProbability.items() if tuple(item[0])[:-1] == tuple(sentence[-self.ngram + 1:])}
            if(len(nextWordDict) == 0):
                nextWord = SENTENCE_END
                sentence.append(UNK)
                continue
            
            nextWord = random.choices(list(nextWordDict.keys()), weights = list(nextWordDict.values()), k = 1)[0]
            sentence.append(nextWord)

    sentence += [SENTENCE_END] * max(self.ngram - 2, 0)
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
    # 6120 IMPLEMENTS
    
    if(sequence[0] == SENTENCE_BEGIN):
        N = len(sequence) - 1
    else:
        N = len(sequence)
    vocabSize = len(self.tokenCount)
    seq_mul = 1
    
    for i in range(self.ngram-1 , N):
        
        if (self.ngram == 1):
            target = sequence[i]
            prob = ( (self.tokenCount[target] + 1) / (sum(self.tokenCount.values()) + vocabSize))
            seq_mul *= 1/prob
        
        else:
            target = sequence[i:i+self.ngram] 
            prob = ( (self.ngramCount[tuple(target)] + 1) / (self.tokenCount[tuple(target[:-1])] + vocabSize))
            seq_mul *= 1/prob
    
    perplexity = seq_mul ** (1/N)
    
    return perplexity

  
# not required
if __name__ == '__main__':
  print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")









