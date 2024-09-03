# Use arbitrary IP for local execution
ip1=127.0.0.1
export MASTER_ADDR=$(hostname)
python3 main_eval.py --nnodes 1 --ntasks 2 --ngpus 0 --ip_address $ip1