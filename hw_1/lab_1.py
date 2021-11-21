# LEC_2
# Right / Left Division
# 4x - 2y + 6z = 8
# 2x + 8y + 2z = 4
# 6x + 10y + 3z = 0

import numpy as np

A = np.array(
  [
    [4,-2,6],
    [2,8,2],
    [6,10,3]
  ]
)
B=np.array(
    [
      [8],[4],[0]
    ]
)
coefficients = np.linalg.inv(A).dot(B)
print(coefficients)