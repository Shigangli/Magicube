import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

order = ['V2,L16R8,S0.7', 'V8,L16R8,S0.7', 'V2,L8R8,S0.7', 'V8,L8R8,S0.7',
         'V2,L8R4,S0.7', 'V8,L8R4,S0.7', 'V2,L4R4,S0.7', 'V8,L4R4,S0.7',
         'V2,L16R8,S0.9', 'V8,L16R8,S0.9', 'V2,L8R8,S0.9', 'V8,L8R8,S0.9',
         'V2,L8R4,S0.9', 'V8,L8R4,S0.9', 'V2,L4R4,S0.9', 'V8,L4R4,S0.9']

#sns.color_palette("Blues", as_cmap=True)
spmm_abl_data = pd.read_csv('spmm_abl_study.csv')
sns.set(rc={"lines.linewidth": 0.5})
sns.set(rc={'figure.figsize':(15, 5)})
g = sns.barplot(data=spmm_abl_data, x="configs", y="TOP/s", hue="opts", order=order, palette="Blues_d")
plt.xticks(rotation=20)
g.tick_params(labelsize=8)
g.set(ylim=(0, 40))
g.figure.savefig('./figs/Figure11.pdf')

