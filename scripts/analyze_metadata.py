import pandas as pd
import numpy as np



# Jurkat + Nematode Images
df = pd.read_table("https://data.broadinstitute.org/bbbc/"
                   "BBBC048/Ground_truth.lst", header=None,
                   names=["id", "cycle", "path"])


# Kaggle Data Science Bowl 2018
df = pd.read_csv("kag_data_bowl_1.csv")  # Training
df = pd.read_csv("kag_data_bowl_2.csv")  # Testing Part I
df = pd.read_csv("kag_data_bowl_3.csv")  # Testing Part II


# Kaggle Human Protein Atlas Classification
df = pd.read_csv("kag_hpa_2019_train.csv")  # Training
for i in range(28):
    df[str(i)] = 0
for idx in range(len(df)):
    labels = np.array(df.loc[idx, "Target"].split(" ")).astype("int")
    for label in labels:
        df.loc[idx, str(label)] = 1
labels_sum = df.sum().drop(["Id", "Target"])
label_names = """nucleoplasm, nuclear membrane, nucleoli, nucleoli fibrillary center, nuclear speckles, nuclear bodies, ER, Golgi apparatus, peroxisomes, endosomes, lysosomes, intermediate filaments, actin filaments, focal adhesion sites, microtubules, microtubule ends, cytokinetic bridge, mitotic spindle, microtubule organizing center, centrosome, lipid droplets, plasma membrane, cell junctions, mitochondria, aggresome, cytosol, cytoplasmic bodies, rods & rings"""
name_map = dict(zip([str(i) for i in range(28)], label_names.split(", ")))
labels_sum.rename(name_map, axis=1, inplace=True)


# Kaggle HPA Single Cell Classification
df = pd.read_csv("kag_hpa_singlecell_2021_train.csv")
for i in range(19):
    df[str(i)] = 0
for idx in range(len(df)):
    labels = np.array(df.loc[idx, "Label"].split("|")).astype("int")
    for label in labels:
        df.loc[idx, str(label)] = 1
labels_sum = df.sum().drop(["ID", "Label"])
# label_names = """nucleoplasm, nuclear membrane, nucleoli, nucleoli fibrillary center, nuclear speckles, nuclear bodies, ER, Golgi body, intermediate filaments, actin, microtubules, mitotic spindle, centrosome, plasma membrane, mitochondria, aggresome, cytosol, vesicles and punctate cytosolic patterns, negative"""
# name_map = dict(zip([str(i) for i in range(19)], label_names.split(", ")))
# labels_sum.rename(name_map, axis=1, inplace=True)
# labels_sum.plot(kind="bar", color=labels_sum.index)
# plt.title("HPA 2021 Single Cell Classification")
# plt.xlabel("Labels")
# plt.ylabel("Frequency")
# plt.tight_layout()


# Kaggle Leukemia Classification
df = pd.read_csv("kag_leukemia_val.csv")


# BBBC Human U20S Cells (Out of Focus)
df = pd.read_csv("bbbc_u2os_focus_metadata.csv", index_col=False)
df.rename(columns={"Image_Count_Nuclei": "num_nuclei",
                   "Image_FileName_OrigDAPI": "filename"}, inplace=True)


# BBBC Human MCF7 - Compound Profiling
df = pd.read_csv("bbbc_mcf7_profile_metadata.csv")
df_labels = pd.read_csv(
    "bbbc_mcf7_profile_metadata_1.csv")
new_df = pd.DataFrame(columns=["moa", "count"])
for i in range(len(df_labels)):
    compound = df_labels.loc[i, "compound"]
    conc = df_labels.loc[i, "concentration"]
    moa = df_labels.loc[i, "moa"]

    idx = (df.Image_Metadata_Compound == compound) & \
          (df.Image_Metadata_Concentration == conc)
    if moa not in new_df.moa.tolist():
        row = pd.Series(dtype=object)
        row["moa"] = moa
        row["count"] = len(df[idx])
        new_df = pd.concat([new_df, pd.DataFrame(row).transpose()],
                           ignore_index=True)
    else:
        moa_idx = (new_df["moa"] == moa)
        new_df.loc[moa_idx, "count"] += len(df[idx])
new_df.sort_values(by="count", ascending=False, inplace=True)
new_df.reset_index(inplace=True, drop=True)

# BBBC Human HT29 shRNAi screen
df = pd.read_excel("bbbc_colon_metadata.xls")

all_labels = []
classes = df["class"].value_counts().index
for labels in classes:
    split = labels.split(";")
    for label in split:
        if label not in all_labels:
            all_labels.append(label)
new_df = pd.DataFrame(columns=all_labels)
for i in range(len(df)):
    row = df.iloc[i]
    if row["class"] == "NONE":
        continue
    try:
        split = row["class"].split(";")
        new_row = pd.Series()
        for label in split:
            new_row[label] = 1
        unlabels = [i for i in all_labels if i not in split]
        for unlabel in unlabels:
            new_row[unlabel] = 0
        new_df = new_df.append(new_row, ignore_index=True)
    except:
        print(row["class"])


# Cell-IDR idr0097 Human Protein GFP-Tagging
names = ['Plate', 'Well', 'Well Number', 'Organism', 'Cell Type', 'Gene', 'Channels', 'Has Phenotype', 'location_1', 'location_2', 'location_3', 'Plate Name', 'Well Name']
to_remove = [4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20]
col_indices = [i for i in range(26) if i not in to_remove]
df = pd.read_csv("cell_idr0097_annotations.csv", index_col=False,
                 skiprows=1, header=None, names=names, usecols=col_indices)
sum_labels = pd.concat([df.location_1.value_counts(),
                        df.location_2.value_counts(),
                        df.location_3.value_counts()])
sum_labels = sum_labels.groupby(by=sum_labels.index).sum()


# Cell-IDR
df = pd.read_csv("https://raw.githubusercontent.com/IDR/idr0078-mattiazziusaj-endocyticcomp/ec4fc5e8697b4bb9ee0158870149def21fe9d2dd/screenA/idr0078-screenA-annotation.csv")



# BDGP, Berkeley Drosophila Genome Project
df = pd.read_csv("bdgp_annotations.csv", header=None, names=[
    "gene", "gene_name", "gene_database", "field_num", "label"])



# Recursion RxRx1
df = pd.read_csv("rec_rxrx1_metadata.csv")

# Recursion RxRx2
df = pd.read_csv("rec_rxrx2_metadata.csv")

# Recursion RxRx19a
df = pd.read_csv("rec_rxrx19a_metadata.csv")

# Recursion RxRx19b
df = pd.read_csv("rec_rxrx19b_metadata.csv")
