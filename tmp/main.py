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


# a = np.arange(10).reshape(2, -1).tolist()
# z = np.arange(10, 20).reshape(2, -1).tolist()
# w = np.arange(20, 30).reshape(2, -1).tolist()
# b = [a, z, w]

# # print(a)
# # print(b)

# c, v, x = b
# d = b

# print(c, type(c))
# print(v, type(v))
# print(x, type(x))
# print(d, type(d))


# grads = [np.zeros_like((2, 1)), np.zeros_like((2, 1)), np.zeros_like((2, 1))]
# print(*grads)


# a = np.arange(12).reshape(3, -1)
# print(a)
# print(np.sum(a, axis=0))
# print(np.sum(a, axis=1))

# def test(a, b, c):
#     print(a)
#     print(b)
#     print(c)

# test_arr = [1, 2, 3]
# test(*test_arr)

# b = np.ones((2, 3, 2))
# print(b)
# print()

# print(b[:, 0, :].shape)

# print(np.empty((1, 2), dtype="int"))

# a = [1, 2, 3]
# b = [i + 1 for i in a]
# c = [i + 2 for i in a]

# print(a)
# print(b)
# print(c)

# grads = [0, 0, 0]

# for i in np.array([a, b, c]):
#     print(i)
#     grads[0] += i

# print(grads)


# e = [1, 2, 3] + [1, 2, 3]
# print(e)

print(np.random.randn(1))

print(1 % 99)