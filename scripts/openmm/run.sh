#BSUB -q cpuqueue
#BSUB -o %J.stdout
#BSUB -R "rusage[mem=1]"
#BSUB -W 23:00
#BSUB -n 32

python test.py