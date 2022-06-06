import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#order = ['L4-R4', 'L8-R4', 'L12-R4', 'L16-R4', 'L8-R8', 'L16-R8', 'L16-R16']
spmm_pres_data = pd.read_csv('spmm_pres.csv')
sns.set(rc={"lines.linewidth": 0.5})
sns.set(rc={'figure.figsize':(15, 5)})
#g = sns.barplot(data=spmm_pres_data, x="configs", y="TOP/s", hue="pres", palette="Blues_d", hue_order=order)
g = sns.barplot(data=spmm_pres_data, x="configs", y="TOP/s", hue="pres", palette="Blues_d")
plt.xticks(rotation=20)
g.tick_params(labelsize=8)
g.set(ylim=(0, 45))
g.figure.savefig('./figs/Figure12.pdf')

