import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sddmm_abl_study_data = pd.read_csv('sddmm_abl_study.csv')
sns.set(rc={"lines.linewidth": 0.5})
sns.set(rc={'figure.figsize':(15, 5)})
g = sns.barplot(data=sddmm_abl_study_data, x="configs", y="TOP/s", hue="pres", palette="Blues_d")
plt.xticks(rotation=20)
g.tick_params(labelsize=8)
g.set(ylim=(0, 40))
g.figure.savefig('./figs/Figure13.pdf')

