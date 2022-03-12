import os

home_dir = "/ferrero/stan_data/"
dir_names = os.listdir(home_dir)

for subdir in dir_names:
    os.chdir(home_dir)
    os.chdir(subdir)
    os.system("find . -depth -name '*.zip' -exec /usr/bin/unzip -n {} \; -exec rm {} \;")
