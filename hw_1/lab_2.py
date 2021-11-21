# LEC_2
# Inverse of a matrix:
import numpy as np
A = np.array(
  [
    [2,1,4],
    [4,1,8],
    [2,-1,3]
  ]
)
B = np.linalg.inv(A)
I = np.dot(A, B)
print(I)