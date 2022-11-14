import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import signal





data = pd.read_pickle('ETHUSDT15M.pkl').loc['2022-11-09':'2022-11-10']


#['2022-11-09':'2022-11-10']
#['2019-04-17':'2019-04-18']
#['2019-04-22':'2019-04-23']

data.reset_index(inplace=True)
dataFiltered = gaussian_filter1d(data.Close, sigma=2)
tMax = signal.argrelmax(dataFiltered)[0]
tMin = signal.argrelmin(dataFiltered)[0]

plt.plot(data.Close, '-')
plt.plot(dataFiltered, '--')

plt.plot(tMax, dataFiltered[tMax], 'o', mfc= 'none', label = 'max')
plt.plot(tMin, dataFiltered[tMin], 'o', mfc= 'none', label = 'min')

plt.show()