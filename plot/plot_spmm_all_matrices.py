import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


spmm_data = pd.read_csv('spmm_all_matrices.csv')
##print(spmm_data)
#
#fgrid = sns.FacetGrid(spmm_data, col="vecLen", row="dimN")
#fgrid.map_dataframe(sns.boxplot, x="Sparsity", y="", data=spmm_data)
#
#
#fgrid.figure.savefig('test.pdf')
sns.set(rc={"lines.linewidth": 0.5})
g = sns.catplot(x="sparsity", y="speedup",
                hue="algs", col="V", row="N", fliersize=3,
                data=spmm_data, kind="box",
                height=4, aspect=1.6)
g.set(ylim=(0.0, 3.0))
plt.axhline(1.0, linestyle='--', linewidth=2.7, color='blue')
g.figure.savefig('./figs/Figure14.pdf')
