if [ ! -d Excels ]; then
    mkdir Excels
fi
if [ ! -d Figures ]; then
    mkdir Figures
fi
python perf_model.py --log-file-name ../DRIM-ANN/build/output.txt --dpu-amount 2543 --host-thread-amount 64
