import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib



font = {'family' : 'serif',
        #'weight': 'bold' ,
        'size'   : 10}

matplotlib.rc('font', **font)

labelFont = {'family' : 'serif',
        #'weight': 'bold' ,
        'size'   : 10}




x = np.arange(-2.0, 6.0, 0.01)
y = 0.2 * np.exp(x) + 0.5 * x - 5.5

plt.plot(x, y, '-k', lw=1)

ax = plt.subplot(111)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks([-2, -1, 1, 2, 3, 4, 5, 6])
ax.yaxis.set_ticks([-10, 10, 20, 30, 40, 50, 60, 70, 80])

#label = ax.set_xlabel('x')
#ax.xaxis.set_label_coords(1.02, 0.640)

plt.show()