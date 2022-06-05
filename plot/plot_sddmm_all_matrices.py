import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


sddmm_data = pd.read_csv('sddmm_all_matrices.csv')
##print(sddmm_data)
#
#fgrid = sns.FacetGrid(sddmm_data, col="vecLen", row="dimN")
#fgrid.map_dataframe(sns.boxplot, x="Sparsity", y="", data=sddmm_data)
#
#
#fgrid.figure.savefig('test.pdf')
sns.set(rc={"lines.linewidth": 0.5})
g = sns.catplot(x="sparsity", y="speedup",
                hue="algs", col="V", row="K", fliersize=3,
                data=sddmm_data, kind="box",
                height=4, aspect=1.6)
g.set(ylim=(0.0, 3.0))
plt.axhline(1.0, linestyle='--', linewidth=2.7, color='blue')
g.figure.savefig('./figs/Figure15.pdf')
