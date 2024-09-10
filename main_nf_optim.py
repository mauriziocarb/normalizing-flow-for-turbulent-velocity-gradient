
##################################################################
### IMPORTS
##################################################################

import os,sys,time
from argparse import ArgumentParser
import pickle
import numpy as np
from itertools import combinations

import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
to.set_default_dtype(to.float32)

import util_code.distributions as distributions
import util_code.volume_elements as volume_elements
import util_code.IO as IO
from util_code.networks import RealNVP
from util_code.basic_util import to_matrix
from util_code.main_util import init_parallel, init_variables, init_model



##################################################################
### PARAMETERS
##################################################################

## Training parameters
logf_network_file = None  # state-file from previous training to use for initialization 
                    # (set to None for random initializition)
N_epochs = 80       # Number of training epochs to compute
lr = 3e-5           # Learning rate of the ADAM-optimizer
N_part    = 128     # Number of particles to use per process (reduced number for example data)
N_batches = 10      # Number of batches to split the data into (should be modified to match available memory)
out_dir = "nf_results/" # directory to save results in
write_states_freq = 10  # write out networks every N-th epoch

## Model parameters
N_var = 8
N_up = 1
N_un = N_var-N_up
N_sbins = 8
N_iter_layer = 4
transf_layer = "linear" # alternatively "rational_quadratic"
distribution = distributions.normal
# neural network parameters
channels= { "widths":  [N_un]+[4*N_sbins*N_up]+[N_sbins*N_up], \
            "heights": [N_un]+[4*N_sbins*N_up]+[N_sbins*N_up], \
            "derivatives": [N_un]+[4*(N_sbins+1)*N_up]+[(N_sbins+1)*N_up],\
            "rotation": [N_un]+[64,64]+[N_up], \
            "aff": [N_un]+[64,64]+[N_up]}

## Data loading parameters
# (supports loading of multiple files from the given dir. as long as they are numbered consecutively, 
#  nfile0 sets the first file to start loading from)
data_dir  = "data/" # base file-directory
file_name = "velocity_gradients_{:03d}.bin" # file-namespace
nfile0    = 1     # ID of the first file to load (useful for skipping a transient period in a sequence of output files)

N_part_per_file    = 1000  # Number of particles saved in each reference data-file (reduced number of example data)
N_steps_per_file   = 500  # Number of timesteps saved in each reference data-file



##################################################################
### CLASSES AND FUNCTIONS
##################################################################

# Trainer class for convient initialization and use of class variables 
# to allow access to general variables across different functions during training
class Trainer():

    def __init__(self, node, ntasks, world_size, ngpus):
        super().__init__()

        self.node = node
        self.ntasks = ntasks
        self.world_size = world_size
        self.ngpus = ngpus
        
        self.N_part  = N_part
        self.N_batches = N_batches
        
        self.N_part_per_file = N_part_per_file
        self.N_steps_per_file  = N_steps_per_file

        self.N_up = N_up
        self.N_un = N_un
        self.N_sbins = N_sbins
        self.N_iter_layer = N_iter_layer
        self.channels = channels

        self.transf_layer = transf_layer
        self.distribution = distribution

        self.lr = lr
        self.N_epochs = N_epochs
        self.logf_network_file = logf_network_file
        self.out_dir = out_dir
        self.write_states_freq = write_states_freq
        
        self.data_dir = data_dir
        self.file_name = file_name
        self.nfile0 = nfile0
        
        self.N_part_per_file = min(world_size*N_part, N_part_per_file) #131072)
        self.N_part_tot      = N_part*world_size

        self.nfiles          = self.N_part_tot//self.N_part_per_file
        self.nranks_per_file = self.N_part_per_file//N_part

    def train(self, task):
        ##################################################################
        ### INITIALIZATION
        ##################################################################


        init_parallel(self,task)
        init_variables(self)
        init_model(self,task)

        self.optimizer = to.optim.Adam(self.log_f.parameters(), lr=self.lr) 
        self.optimizer.zero_grad()

        if(self.rank==0):
            os.makedirs(self.out_dir+"/states/",exist_ok=True)
            os.makedirs(self.out_dir+"/stats/",exist_ok=True)

            # empty logfile
            with open(self.out_dir+"/loss.txt", "w") as f:
                f.write("Epoch\tTime\tLoss\tG-Loss\tCons.\tISO\n")


        ##################################################################
        ### LOADING OF REFERENCE DATA
        ##################################################################

        fn = self.data_dir+self.file_name.format(self.nfile0+self.rank//self.nranks_per_file)
        A  = IO.read_binary_DNS(fn, self.N_part_per_file, self.N_steps_per_file, self.rank, self.nranks_per_file)

        A = A.reshape(-1,3,3)
        tau_eta = to.einsum("sij,sij->s", A, A).mean().to(self.device)
        dist.all_reduce(tau_eta, op=dist.ReduceOp.SUM)
        tau_eta/=self.world_size
        tau_eta = to.sqrt(1/tau_eta).to("cpu")
        if self.rank==0:
            print("Kolmogorov time", tau_eta.item())
        A*=tau_eta

        data_unnorm = A.reshape(-1,9)[:,:self.N_var].float()
        self.N_samples = data_unnorm.shape[0]
        
        mom   = to.zeros([3], device=self.device).detach()
        mom[0] = to.sum(data_unnorm**2).to(self.device).detach()
        mom[1] = to.sum(data_unnorm**3).to(self.device).detach()
        mom[2] = to.mean(data_unnorm**4).to(self.device).detach()
        dist.all_reduce(mom, op=dist.ReduceOp.SUM)
        mom /= (1.*self.world_size)
        mom[0] = 1. #1/tau_eta**2
        mom[1] = 0.
        mom[2] = 0. #Q and R avg

        if self.transf_layer=="rational_quadratic":
            data   = (1/(1+to.exp(-data_unnorm.double()))).float()
        elif self.transf_layer=="linear":
            data = data_unnorm

        self.batch_size = self.N_samples//self.N_batches
        train_loader = to.utils.data.DataLoader(data, batch_size=self.batch_size)



        ##################################################################
        ### TRAINING
        ##################################################################

        losses = []
        for epoch in range(self.N_epochs):
            #############################
            ## EPOCH OPTIMIZATION

            time0 = time.time() 
            epoch_loss   = 0
            epoch_iso    = 0
            g_epoch_loss = 0
            epoch_cons   = 0
            pdf = np.zeros((1024,))

            for batch_index, training_sample in enumerate(train_loader):
                g_mom = to.zeros([3], device=self.device)

                inp = training_sample.to(self.device)
                x, log_prob = self.log_f.log_probability(inp, self.transf_layer)
                loss = -log_prob.sum()

                target = to.normal(0,1,inp.shape,device=self.device)
                with to.no_grad():
                    R = to_matrix(target.detach())
                    S = (R + to.transpose(R, -1,-2)).detach()
                    _, V = to.linalg.eigh(R)
                    B = to.einsum("sik,skl,sjl->sij", V, to_matrix(inp), V).reshape(-1,9)[:,:self.N_var]
                    _, log_probR = self.log_f.log_probability(B, self.transf_layer)
                    loss_iso = to.sum((log_prob - log_probR)**2)
                    epoch_iso += loss_iso.detach()

                    v = (log_prob-log_probR).cpu().numpy()
                    aux, bins = np.histogram(v, bins=1024, range=[-4,4], density=False)
                    aux = to.tensor(aux, device=self.device)
                    dist.reduce(aux, 0, op=dist.ReduceOp.SUM)
                    pdf += aux.cpu().numpy()

                    consistency, _ = self.log_f.rsample(x, self.transf_layer)
                    consistency = to.sum((consistency-inp)**2)
                    epoch_cons += consistency.detach()
                
                if self.transf_layer=="rational_quadratic":
                    target = 1/(1+to.exp(-target.double())).float()
                g_x, g_log_prob = self.log_f.rsample(target, self.transf_layer)
                if self.transf_layer=="rational_quadratic":
                    y = g_x.double()
                    g_x = to.log(y/(1. - y)).float()
                g_A = to_matrix(g_x)
                g_mom[0] = to.einsum("sij,sij->s", g_A, g_A).mean()
                g_mom[1] = to.einsum("sij,sji->s", g_A, g_A).mean()
                g_mom[2] = to.einsum("sij,sjk,ski->s", g_A, g_A, g_A).mean()

                dist.all_reduce(g_mom, op=dist.ReduceOp.SUM)
                g_mom /= (1.*self.world_size)
                g_loss = to.sum((g_mom-mom)**2)
                g_epoch_loss += g_loss.detach()
                
                loss += g_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.detach()
                self.log_f.zero_grad(set_to_none=True)
                del log_prob, loss, x

            #############################
            ## OUTPUT AND LOGGING

            dist.reduce(epoch_loss, 0, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss.item()/(1.*self.world_size*self.N_samples)

            dist.reduce(g_epoch_loss, 0, op=dist.ReduceOp.SUM)
            g_epoch_loss = g_epoch_loss.item()/(1.*self.world_size*self.N_samples)
            dist.reduce(epoch_cons, 0, op=dist.ReduceOp.SUM)
            epoch_cons = epoch_cons.item()/(1.*self.world_size*self.N_samples)

            dist.reduce(epoch_iso, 0, op=dist.ReduceOp.SUM)
            epoch_iso = epoch_iso.item()/(1.*self.world_size*self.N_samples)
            del g_log_prob, g_loss, g_x, g_mom, target, consistency, loss_iso

            if self.rank==0:
                print("Epoch {} ({} s)\t".format(epoch, float(time.time()-time0)),"Losses: ", epoch_loss, "\t", g_epoch_loss, "\t", epoch_cons, "\t", epoch_iso)
                sys.stdout.flush()
                with open(self.out_dir+"/loss.txt", "a") as fh:
                    fh.write(("%d\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\n" % (epoch, float(time.time()-time0), epoch_loss, g_epoch_loss, epoch_cons, epoch_iso)))
                if epoch%self.write_states_freq==0:
                    IO.save_model(self.log_f,self.out_dir+"/states/f_state_epoch_{}.pt".format(epoch))

                    bins = .5*(bins[1:]+bins[:-1])
                    pickle.dump([pdf,bins], open((self.out_dir+"/stats/iso_error_pdf_epoch%d.bin" % epoch), "wb"))

        dist.barrier()
        return



##################################################################
### MAIN METHOD
##################################################################

if __name__=="__main__":
    #run parameters
    parser = ArgumentParser()
    parser.add_argument("--ntasks", default=1, type=int, help="number of tasks per node (same as number of GPUs if ngpus>0)")
    parser.add_argument("--ngpus",  default=1, type=int, help="number of gpus per node")
    parser.add_argument("--nnodes", default=1, type=int, help="number of nodes")
    parser.add_argument("--ip_address", required=True, type=str, help="IP address of the master node")
    args = parser.parse_args()
    assert args.nnodes > 0
    assert args.ntasks > 0
    if args.ngpus > 0:
        assert to.cuda.is_available()
        assert args.ngpus == args.ntasks
        args.device = "cuda:"

    # are we on a cluster? If not, default node 0
    node = os.environ.get("SLURM_NODEID")
    node = int(node) if node is not None else 0
    # in total ntasks tasks per node
    world_size = args.ntasks*args.nnodes
    assert node < args.nnodes
    print("Here is node", node+1, "/", args.nnodes)
    #set address of the master node
    print("Master ip_address is", args.ip_address)
    os.environ["MASTER_ADRR"] = args.ip_address
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(world_size)

    T = Trainer(node, args.ntasks, world_size, args.ngpus)
    mp.spawn(T.train, nprocs=args.ntasks)
