import numpy as np	
import time
size = 5
a = np.reshape(np.arange(size*size),(size, size))
b = np.reshape(np.arange(size*size),(size, size))
c = a @ b
print(c)
