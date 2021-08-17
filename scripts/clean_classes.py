import json
import glob

import pandas as pd

annotations_dir = "/home/stan/cytoimagenet/annotations/"
data_dir = '/ferrero/stan_data/'


def over_populated():
    """Return list of labels containing greater than 1000 samples.
    """
    labels = []
    for file in glob.glob(f"{annotations_dir}classes/*.csv"):
        df = pd.read_csv(file)
        if len(df) > 1000:
            labels.append(file.split("classes\\")[-1].replace(".csv", ""))

    return labels


def downsample(df, cols=("name", "organism", "cell_type", "cell_visible")) -> pd.DataFrame:
    """Return metadata dataframe <df> downsampled to exactly 1000 rows if
    contains >1000 rows. Else, return <df> as is.

    Method of down-sampling:
        - Stratified sampling via cols
    """
    if len(df) <= 1000:
        return df

    # Stratified Sampling
    frac_to_sample = 1000 / len(df)
    df_new = df.groupby(cols, dropna=False, group_keys=False).sample(frac=frac_to_sample)

    # If still over 1000, randomly sample 1000 exactly
    if len(df_new) > 1000:
        print(f"{1000} randomly sampled from {len(df_new)}!")
        return df_new.sample(n=1000)
    elif len(df_new) == 1000:
        return df_new
    else:  # if less than 1000, randomly sample enough to get back to 1000
        print(f"{1000 - len(df_new)} randomly sampled to supplement "
              f"initial sampling!")
        sup = df[~df["idx"].isin(df_new["idx"])].sample(n=1000 - len(df_new))
        return pd.concat([df_new, sup])


def check_sample_size(label: str):
    """Checks if class <label> has more than 1000 examples. If so, downsample
    accordingly and save.
    """
    with open(f"{annotations_dir}classes/used_images.json") as f:
        used_indices = json.load(f)

    df = pd.read_csv(f"{annotations_dir}classes/{label}.csv")
    if len(df) <= 1000:
        print(f"{label} has <= 1000 examples!")
    else:
        df_down = downsample(df, cols=["name", "organism", "cell_type", "cell_visible"])
        idx_to_remove = df[~df["idx"].isin(df_down["idx"])].index.tolist()

        for idx in idx_to_remove:
            used_indices.pop(idx, None)

        df_down.to_csv(f"{annotations_dir}classes/{label}.csv", index=False)

    with open(f"{annotations_dir}classes/used_images.json", 'w') as f:
        json.dump(used_indices, f)


if __name__ == "__main__":
    # with open(f"{annotations_dir}classes/used_images.json") as f:
    #     used_indices = json.load(f)
    #
    # labels = check_existing_classes()
    # for label in labels:
    #     df = pd.read_csv(f"{annotations_dir}classes/{label}.csv")
    #     if len(df) <= 1000:
    #         print(f"{label} has <= 1000 examples!")
    #     else:
    #         df_down = downsample(df, cols=["name", "organism", "cell_type", "cell_visible"])
    #         idx_to_remove = df[~df["idx"].isin(df_down["idx"])].index.tolist()
    #
    #         for idx in idx_to_remove:
    #             used_indices.pop(idx, None)
    #
    #         df_down.to_csv(f"{annotations_dir}classes/{label}.csv", index=False)
    #
    # with open(f"{annotations_dir}classes/used_images.json", 'w') as f:
    #     json.dump(used_indices, f)
    pass
