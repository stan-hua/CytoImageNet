import os
import pandas as pd

annotations_dir = "D:/projects/cytoimagenet/annotations/"

df = pd.read_csv(f"{annotations_dir}datasets_info.csv")

files = os.listdir(annotations_dir)

for name in files:
    if sum(df.dir_name.map(lambda x: x in name)) == 0:
        if name == "datasets_info.csv":
            continue
        
        print(f"Invalid Name! {name}")
        new_name = input("New filename: ")
        os.system(f"mv {annotations_dir}{name} {annotations_dir}{new_name}")
        print("\n\n")
