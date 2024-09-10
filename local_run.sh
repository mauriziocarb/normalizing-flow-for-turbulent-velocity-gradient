export MASTER_ADDR=$(hostname)
python3 main_eval.py --nnodes 1 --ntasks 2 --ngpus 0 --ip_address 127.0.0.1 # Use localhost IP for local execution