#!/bin/bash
DATADIR='results_artifact'
DATAFILE="$DATADIR.csv"

ITERS=(1 2 5 10 20 50 100 200 500 1000 2000)
SIZES=(1 2 4 8)
SUBMISSIONS=(baseline CNP graph vulkan)
MEASUREMENTS=(sub exe)
RUNS=100

#ITERS=(1)
#SIZES=(1)
#SUBMISSIONS=(baseline)
#MEASUREMENTS=(sub)
#RUNS=2

HEADER='EMPTY'
# write header line
for submission in ${SUBMISSIONS[*]}; do
  for size in ${SIZES[*]}; do
    for iter in ${ITERS[*]}; do
      for measurement in ${MEASUREMENTS[*]}; do
        HEADER=$([ "$HEADER" == 'EMPTY'  ] \
        && echo $submission':'$measurement':i'$iter':s'$size \
        || echo $HEADER' '$submission':'$measurement':i'$iter':s'$size
        )
      done
    done
  done
done
echo "$HEADER" > "$DATAFILE"

# collect files in columns
for run in `seq 1 $RUNS`; do
  for submission in ${SUBMISSIONS[*]}; do
    for size in ${SIZES[*]}; do
      for iter in ${ITERS[*]}; do
        for measurement in ${MEASUREMENTS[*]}; do
          file='./'$DATADIR'/'$submission'/'$measurement'_i'$iter'_s'$size'.txt'
          echo -n `sed "$run"'q;d' $file`' ' >> $DATAFILE
        done
      done
    done
  done
  echo >> $DATAFILE
done
exit

echo -n $submission'_'$measurement'_i'$iter'_s'$size > "$DATAROOT".csv


