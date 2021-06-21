import dask.dataframe as dd
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    plot_dir = "/home/stan/cytoimagenet/figures/classes/"
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

cols = ["organism", "cell_type", "cell_component", "phenotype", "gene", "sirna", "compound", "microscopy"]

df = dd.read_csv(f"{annotations_dir}clean/*_metadata.csv",
                 dtype={"organism": "object",
                        "cell_type": "object",
                        "cell_component": "object",
                        "phenotype": "object",
                        "gene": "object",
                        "sirna": "object",
                        "compound": "object",
                        "microscopy": "object",
                        })

sns.set_style("white")
for i in cols:
    try:
        fig = plt.figure()
        plt.style.use("seaborn-colorblind")
        plt.title(i)
        plt.xlabel("Count")
        df.loc[:, i].value_counts().compute().plot(kind='barh',
                                                   logx=True,
                                                   colormap="Set3")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{i}_classes.png")

        num_classes = (df.loc[:, i].value_counts() >= 200).sum().compute()

        print(f"{i} has {num_classes}")

    except:
        print(f"{i} failed!")
