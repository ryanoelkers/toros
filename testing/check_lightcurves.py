import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats

star_list = pd.read_csv(
    "C:\\Users\\ryanj\\OneDrive - The University of Texas-Rio Grande Valley\\Research\\TOROS\\master\\"
    + 'FIELD_0e.001_star_list.txt', delimiter=' ',
    header=0)

lc1 = pd.read_csv("C:\\Users\\ryanj\\OneDrive - The University of Texas-Rio Grande Valley\\Research\\TOROS\\lc\\003593.lc", sep=' ')

lc1['ph'] = (lc1.jd - lc1.jd.to_numpy()[0]) / 0.73711 % 1
plt.scatter(lc1[lc1.mag > 5].ph, lc1[lc1.mag > 5].mag, c='k')
plt.scatter(lc1[lc1.mag > 5].ph, lc1[lc1.mag > 5].cln, c='r')
plt.gca().invert_yaxis()
plt.show()
lc2 = pd.read_csv("C:\\Users\\ryanj\\OneDrive - The University of Texas-Rio Grande Valley\\Research\\TOROS\\lc\\001384.lc", sep=' ')

lc2['ph'] = (lc2.jd - lc2.jd.to_numpy()[0]) / 0.336775 % 1
plt.scatter(lc2[lc2.mag > 5].ph, lc2[lc1.mag > 5].mag, c='k')
plt.scatter(lc2[lc2.mag > 5].ph, lc2[lc1.mag > 5].cln, c='r')
plt.gca().invert_yaxis()
plt.show()

lc3 = pd.read_csv("C:\\Users\\ryanj\\OneDrive - The University of Texas-Rio Grande Valley\\Research\\TOROS\\lc\\000281.lc", sep=' ')

lc4 = pd.read_csv("C:\\Users\\ryanj\\OneDrive - The University of Texas-Rio Grande Valley\\Research\\TOROS\\lc\\037378.lc", sep=' ')


print('hold')