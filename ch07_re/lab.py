import numpy as np

index_size = 6
a = np.arange(20).reshape(10, -1).astype("f")
b = np.arange(0, 120, 10).reshape(index_size, -1)

rng = np.random.default_rng()

# print(a, end="\n")
# print(b)

# data_size = a.shape[0]
# index = np.arange(index_size)

# print(f"サイズ: {data_size}")
# print(f"インデックス: {index}")

# # 書き込まれる側, 書き込み先, 書き込むデータ
# np.add.at(a, index, b)
# print(a)

print(rng.integers(0, 3, size=(3, 3), endpoint=True))