import numpy as np
import copy
import collections

h = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) # (2, 3)
w = np.arange(12).reshape(3, -1) # (3, 4)
# print(w)

idx = np.array([0, 1])

# print(w[:, idx])print(w)
# print(h @ w[:, idx])
# print(w[:, 0])
# print()
# print(h[0] @ w[:, 0])
# print(np.sum(h * w[:, idx], axis=1))

# c = (1, 10)
# d, e = c
# print(d, e)

# dw = np.zeros((6, 3))
# dh = np.ones((3, 3))
# idx = np.array([0, 1, 0])

# np.add.at(dw, idx, dh)
# print(dw)

# for i in np.arange(0.1, 1, 0.1):
#     print(f"{i}: {-np.log(i)}")

# a = [0, 0, 1, 0]
# counter = collections.Counter(a)
# print(counter[0])
# print(len(counter))

# print(counter[0, 1])

# 元の配列
# a = np.arange(5)
# print("a:", a)

# # ビューを作成（メモリは共有している）
# b = a[::2]
# print("b (view):", b)

# c = np.copy(b)

# d = copy.deepcopy(b)
# # 元配列を書き換え
# a[0] = 99
# print("\nAfter modifying a[0] = 99")
# print("a:", a)
# print("b (view):", b)
# print("c (np.copy):", c)
# print("d (deepcopy):", d)


# ランダムチョイス
# a = np.random.choice(3, size=3, replace=False, p=[0.2, 0.5, 0.3])
# print(a)

a = np.array([np.random.random() for i in range(100)])
b = np.sum(a)

c = a / b
d = np.sum(c)
print(c)
print(np.sum(c))