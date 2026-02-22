import cupy as cp

def cross_entropy(x, t):
    loss = t * cp.log(x)
    loss = -cp.sum(loss)
    return loss

x = cp.linspace(0.1, 1, 10)
t = cp.ones_like(x)
print(x)
print(t)
print(x.shape, t.shape)

loss = cross_entropy(x, t)
print(loss)
# print(cp.log(x))