import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
#data = np.random.randint(10, 30, 50)
#data = np.array([0.01021, 0.011004, 0.010868, 0.011072, 0.011223, 0.010629, 0.011198, 0.01027, 0.010863, 0.010955, 0.010587, 0.011011, 0.010517, 0.011234, 0.011296, 0.010959])
data = np.array([0.00621, 0.007004, 0.010868, 0.011072, 0.011223, 0.010629, 0.011198, 0.01027, 0.010863, 0.010955, 0.010587, 0.011011, 0.010517, 0.011234, 0.011296, 0.010959])

meanv = np.mean(data)
inter = st.norm.interval(alpha=0.95, loc=np.mean(data), scale=st.sem(data))
print("interval: ", inter, "mean: ", meanv)

