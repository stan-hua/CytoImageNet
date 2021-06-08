cp -f D:/projects/cytoimagenet/annotations/* M:/home/stan/annotations
cp -f D:/projects/cytoimagenet/scripts/* M:/home/stan/scripts

rm -r D:/projects/cytoimagenet/scripts/*
rm -f D:/projects/cytoimagenet/annotations/*

cp -s M:/home/stan/annotations/* D:/projects/cytoimagenet/annotations
cp -s M:/home/stan/scripts/* D:/projects/cytoimagenet/scripts

for f in ./*; do if test -h "$f"; then echo "$f is a symlink"; else echo "$f is not a symlink"; fi; done
