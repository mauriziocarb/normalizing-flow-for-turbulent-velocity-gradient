# Tailor-designed models for the turbulent velocity gradient through normalizing flow
Implementation and evaluation of the normalizing flow model in the paper:

> [Tailor-designed models for the turbulent velocity gradient through normalizing flow](https://arxiv.org/abs/???). 

## Steps for replicating the results in the paper
1. Run "main_nf_optim.py" for training the normalizing flow model to learn the single-time velocity gradient PDF of your data.
2. Run "main_traj_optim.py" and load your previously trained normalizing flow model to train the separate multi-time statistics model by matching conditional time derivatives and correlations of your data.
3. Run "main_eval.py" with the two trained models to generate trajectories for evaluation.

## Including the training data
- A sample of the dataset used for training to obtain the results shown in the paper can be downloaded from ???.
- Parallelized data-loading is handled by the function "read_bindary_DNS" in "util_code/IO.py". When using your own data, this function can be adapted for loading training data with different layouts or fileformats from the binary data used in the paper.

## Notes on running the code
- Relevant parameters for the three main programs are set at the top of each file with explanations given as comments for the meaning and possible values of each parameter.
- All main-files are designed to be launched in parallel through MPI. An example script for launching one of the files via the Slurm workload manager is given in the slurm_template.sh file.
- The modular three-file structure of the main code was chosen to allow evaluation and diagnostics of the individual outputs of each step before continuing with the next one.
  For example one may want to assure that the training in step 1. indeed converged towards a satisfying single-time PDF, before continuing to train the multi-time statistics.
  Successive files have respective parameters for loading the trained model-states from the previous steps. Network-parameters of the loaded networks naturally have to match the values set in previous programs that were used to train them.

## Licence
???
