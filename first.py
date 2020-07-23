import numpy as np
import matplotlib.pyplot as plt

# np.random.randint(low,high,num) -> 范围内的n个整数（维度）
ran_int = np.random.randint(0,10,(3,4))
print('ran_int: ',ran_int)
# np.random.random() -> 生成0-1的随机浮点数
ran_0to1 = np.random.random((3,4))
print('ran_0to1: ',ran_0to1)
# np.random.normal -> 正态分布
ran_nor = np.random.normal(loc=0, scale=1)
print(ran_nor)

a = np.arange(0,10).reshape(2,5)
print(a)
print(a.shape)

another_a = np.arange(0,27).reshape(3,3,3)
print(another_a)

# 获取数组维度
print(another_a.ndim)
# 只关心行或列
a = a.reshape(5,-1)
print(a)
a = a.reshape(-1,5)
print(a)

print(another_a[2,1,1])

# 切片操作
print(another_a[1])

demo001 = np.ones([4,4],dtype=int)
demo002 = np.eye(4,dtype=int)
print(demo001 - demo002)

dot1 = np.array([[1,2,3],[4,5,6]])
dot2 = np.array([[1,2],[3,4],[5,6]])
print(dot1.dot(dot2))

trans = np.arange(1,13).reshape(3,4).T
print(trans)
print(trans.sum())
output = trans.argmax()
print(output)

# Sigmoid函数图形


def Sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


plot_x = np.linspace(-10, 10, 200)
plot_y = Sigmoid(plot_x)
plt.plot(plot_x, plot_y)
plt.show()


