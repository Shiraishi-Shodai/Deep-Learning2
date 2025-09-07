import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
from common.trainer import Trainer
from common.optimizer import Adam
# from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot
import numpy as np

window_size = 1
hidden_size = 5
batch_sizze = 3
max_epoch = 5

text = "You say goodbye and I say hello."
corpus , word_to_id, id_to_word = preprocess(text) # (8,)(6,)(6,)
contexts, target = create_contexts_target(corpus, window_size=1) # (6, 2), (6,)
print(contexts.shape, target.shape)

convert_one_hot(contexts, len(word_to_id.keys()))
convert_one_hot(target, len(word_to_id.keys()))
# print(corpus)
# print(id_to_word)