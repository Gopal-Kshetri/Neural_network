import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(np.dot(softmax_output, target_output)))
print(loss)