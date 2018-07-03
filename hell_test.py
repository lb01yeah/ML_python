import numpy as np
from matplotlib import pyplot as plt
print("hell")
print(['he']*4)
print(3 in [1,2,3])

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
print(b)
c = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print('\n',c.shape)

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y, label='$sin(x)$', color='red', linewidth=3)
#plt.plot(x, y, 'b-',label='$sin(x)$', linewidth=3)
plt.ylim(-0.75, 1)
plt.xlabel("label:x")
plt.ylabel("sin(x)")
plt.show()

