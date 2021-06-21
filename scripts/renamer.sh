#find . -type f -name "Zb_*" -exec /usr/bin/rename  Zb_ '' {} \;

find /ferrero/stan_data/idr0072 -depth -name '*.flex' -exec /valr/alexlu/PMCode/bin/PMFlexToTiff -f {} `echo {} | sed 's/.flex/.tif/'`\;
