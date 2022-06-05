import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

spmm_abl_data = pd.read_csv('spmm_abl_study.csv')
sns.set(rc={"lines.linewidth": 0.5})
sns.set(rc={'figure.figsize':(15,3.5)})
g = sns.histplot(data=spmm_abl_data, x="configs", y="TOP/s", hue="opts", multiple="dodge", shrink=.8)

g.set(ylim=(0, 40))
g.figure.savefig('./figs/Figure11.pdf')
