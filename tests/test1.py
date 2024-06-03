import numpy as np

# 创建两个 1 维数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 沿第一个轴连接
# c = np.concatenate((a, b))
c = np.concatenate((a, b))
print(c)  # 输出: [1 2 3 4 5 6]

# 创建两个 2 维数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# 沿第一个轴连接
c = np.concatenate((a, b), axis=0)
print(c)
# 输出:
# [[1 2]
#  [3 4]
#  [5 6]]
