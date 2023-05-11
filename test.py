import numpy as np	
a = np.ones((3, 2))*2
b = np.ones((3, 1))*3
# print(a, b)
c = np.reshape(np.arange(12), (3,2,2)) # [0 1 {2 3} 4 5]
# [0,1,2,3,4,5,6,7,8,9,10,11,12]
print(c, c[1]) # 2,3