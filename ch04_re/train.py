import sys
sys.path.append("..")
import numpy as np
from common.util import create_contexts_target, preprocess
from cbow import CBOW

def main():
    text = "You say goodby and I say hell."
    window_size = 1

    corpus, word_to_id, id_to_word = preprocess(text)
    contexts, targets = create_contexts_target(corpus, window_size) 

    vocab_size = len(word_to_id)
    H = 3

    W_in = np.random.rand(vocab_size, H).astype("f")
    W_out = np.random.rand(vocab_size, H).astype("f")

    power = 0.75
    sample_size = 3


if __name__ == "__main__":
    main()