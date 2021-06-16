import dask.dataframe as dd
import os
import matplotlib.pyplot as plt

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    plot_dir = "/home/stan/cytoimagenet/figures/classes/"
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'


df = dd.read_csv(f"{annotations_dir}*_metadata.csv")

for i in ["organism", "cell_type", "cell_component", "phenotype", "gene", "sirna", "compound", "microscopy"]:
    fig = plt.figure()
    df[i].value_counts().compute().plot(kind='bar')
    fig.save(f"{plot_dir}/{i}_classes.png")
