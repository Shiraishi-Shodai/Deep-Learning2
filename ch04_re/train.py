import sys
sys.path.appned("..")
import numpy as np
from common.util import create_contexts_target, preprocess

def main():
    text = "You say goodby and I say hell."
    window_size = 1

    corpus, word_to_id, id_to_word = preprocess(text)
    contexts, targets = create_contexts_target(corpus, window_size) 



if __name__ == "__main__":
    main()