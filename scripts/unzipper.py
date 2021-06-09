import os

home_dir = "/ferrero/stan_data/"
dir_names = os.listdir(home_dir)

for subdir in dir_names:
    os.chdir(home_dir)
    os.chdir(subdir)
    os.system("find . -depth -name '*.zip' -exec /usr/bin/unzip -n {} \; -exec rm {} \;")


def mover(x):
    val_dir = "M:/ferrero/stan_data/kag_leukemia/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data/"
    out_dir = "M:/ferrero/stan_data/kag_leukemia/C-NMC_Leukemia/"
    if x.labels == 1:
        os.rename(val_dir + x.new_names, out_dir + "all/" + x.Patient_ID)
    else:
        os.rename(val_dir + x.new_names, out_dir + "hem/" + x.Patient_ID)
