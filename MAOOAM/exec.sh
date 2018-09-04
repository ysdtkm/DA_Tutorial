#!/bin/bash

set -e

wdir_base="/lustre/tyoshida/shrt/exec"
modeldir="/lustre/tyoshida/prgm/DA_Tutorial/MAOOAM"
storagedir="/lustre/tyoshida/shrt/sync/maooam"

# preparation
word=t`/lustre/tyoshida/repos/python/pythonpath/oneliner/serial`
wdir="${wdir_base}/${word}"
parallel=false

cd ${modeldir}
make clean
rm -f *.pkl *.npy *.html
git commit -a --allow-empty -m "DA_Tutorial/MAOOAM exec.sh auto commit: experiment ${word}"

echo "preparing files at ${wdir}"
rm -rf ${wdir}
if [ ${parallel} = "true" ]; then
  mkdir -p ${wdir}/template
  cp -rf ${modeldir}/* ${wdir}/template/
else
  mkdir -p ${wdir}
  cp -rf ${modeldir}/* ${wdir}/
fi
cd ${wdir}

echo "#!/bin/bash"                                    > tmp.sh
echo "#SBATCH -n 20"                                 >> tmp.sh
echo "#SBATCH -t 04:15:00"                           >> tmp.sh
echo "#SBATCH -J ${word}"                            >> tmp.sh
echo "set -e"                                        >> tmp.sh
if [ ${parallel} = "true" ]; then
  echo "python3 template/wrap_parallel.py ${wdir}"     >> tmp.sh
  echo "cp -r out ${storagedir}/${word}"               >> tmp.sh
else
  echo "make all"                                      >> tmp.sh
  echo "cp out.pdf ${storagedir}/${word}.pdf"          >> tmp.sh
fi

sbatch tmp.sh
