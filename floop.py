import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(1000)

fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, rectangles = ax.hist(x, 50, normed=True)
fig.canvas.draw()
plt.show()