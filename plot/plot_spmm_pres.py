import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

spmm_pres_data = pd.read_csv('spmm_pres.csv')
sns.set(rc={"lines.linewidth": 0.5})
sns.set(rc={'figure.figsize':(15, 5)})
g = sns.barplot(data=spmm_pres_data, x="configs", y="TOP/s", hue="pres", palette="Blues_d")
plt.xticks(rotation=20)
g.tick_params(labelsize=8)
g.set(ylim=(0, 45))
g.figure.savefig('./figs/Figure12.pdf')

