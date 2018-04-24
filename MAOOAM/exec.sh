#!/bin/bash

set -e

wdir_base="/lustre/tyoshida/shrt/exec"
modeldir="/lustre/tyoshida/prgm/DA_Tutorial/MAOOAM"
storagedir="/lustre/tyoshida/data/sync/maooam"

# preparation
word=t`/lustre/tyoshida/repos/python/pythonpath/oneliner/serial`
wdir="${wdir_base}/${word}"

cd ${modeldir}
rm -f *.pkl *.npy
git commit -a --allow-empty -m "MAOOAM/fortran exec.sh auto commit: experiment ${word}"

echo "preparing files at ${wdir}"
rm -rf ${wdir}
mkdir -p ${wdir}
cd ${wdir}

# cp -rf ${modeldir}/* ./template
cp -rf ${modeldir}/* .

echo "#!/bin/bash"                                    > tmp.sh
echo "#SBATCH -n 20"                                 >> tmp.sh
echo "#SBATCH -t 01:15:00"                           >> tmp.sh
echo "#SBATCH -J ${word}"                            >> tmp.sh
echo "set -e"                                        >> tmp.sh
# echo "python3 template/wrap_parallel.py ${wdir}"              >> tmp.sh
echo "make clean all"                                >> tmp.sh
echo "cp out.pdf ${storagedir}/${word}.pdf"          >> tmp.sh

sbatch tmp.sh
