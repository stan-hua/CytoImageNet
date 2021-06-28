#find . -type f -name ".TIF" -exec /usr/bin/rename  .TIF '.tif' {} \;

for x in `find . -depth -type f -name "*.TIF"` ; do mv "$x" "${x%.TIF}.tif"; done


#find /ferrero/stan_data/idr0081 -depth -name '*.flex' -exec /valr/alexlu/PMCode/bin/PMFlexToTiff -f {} `echo {} | sed 's/.flex/.tif/'`\;
#find . -type f -name "3DT1.nii.gz" -exec rename  3DT1.nii.gz 'T1.nii.gz' {} \;
