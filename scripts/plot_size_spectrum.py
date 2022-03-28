import sys
import os
import numpy as np
import glob
import pandas
import matplotlib.pyplot as plt

nom_res = 0.05 * 2

base_path = sys.argv[1]

start_index = int(sys.argv[2])
nfiles = int(sys.argv[3])

csv_files = glob.glob(os.path.join(base_path,'*.csv'))

print(len(csv_files))

esd = []

for f in csv_files[start_index:(start_index+nfiles)]:
    
    #print(f)
    d = pandas.read_csv(f)
    #print(d)
    esd = esd + (list(d.equivalent_diameter))

esd = np.array(esd)*nom_res
print(esd.shape)
bins = np.logspace(-1,1,25)

# the histogram of the data
n, bins = np.histogram(esd, bins)
plt.loglog(bins[1:], n/nfiles, '*-', color='red')
plt.xlim(.1, 10)
plt.ylim(.01, 10000)



plt.xlabel('Equivalent Spherical Diameter (mm)')
plt.ylabel('Number per Image')
plt.title('Histogram of Detected Particle Size')
plt.grid(True)
plt.show()