
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
from util_code.networks import RealNVP, MLP
from util_code.basic_util import to_matrix_extended, gradient, divergence
from util_code.statistics_util import correlations
from util_code.main_util import init_parallel, init_variables, init_model



##################################################################
### PARAMETERS
##################################################################

## Training parameters
logf_network_file = "nf_results/states/f_state_epoch_80.pt" # state-file for the previously trained single-point pdf network (see main_nf_optim.py file)
G_network_file = None  # state-file from previous training to use for initialization of the multi-time stat. network
                       # (set to None for random initializition)
N_epochs = 20          # Number of training epochs to perform
lr   = 1e-4            # Learning rate of the ADAM-optimizer
beta = 1e-5            # Weighting factor for the conditional time derivative loss contribution
out_dir = "traj_results/" # directory to save results in
write_states_freq = 1  # write out networks every N-th epoch

## Integration parameters
dt = 1e-3         # dt of time-integration
N_steps   = 256   # number of timesteps to integrate (has to be equal to or smaller than available data N_files * N_steps_per_file,
                  #                                   reduced number for example data)
N_part    = 128   # Number of particles to use per process (reduced number for example data)
N_batches = 10    # Number of batches to split the given number of particles into (should be modified to match available memory)

## Model parameters
N_var = 8
N_up = 1
N_un = N_var-N_up
N_sbins = 8
N_iter_layer = 4
transf_layer = "linear" # alternatively "rational_quadratic"
distribution = distributions.normal ##_01 #normal_01 #normal #strain
# neural network parameters
channels= { # Single-point pdf network not optimized here (only loaded from a previous nf_optim)
            "widths":  [N_un]+[4*N_sbins*N_up]+[N_sbins*N_up], \
            "heights": [N_un]+[4*N_sbins*N_up]+[N_sbins*N_up], \
            "derivatives": [N_un]+[4*(N_sbins+1)*N_up]+[(N_sbins+1)*N_up],\
            "rotation": [N_un]+[64,64]+[N_up], \
            "aff": [N_un]+[64,64]+[N_up], \
            # Parameters of multi-time network optimized here
            "G": [N_var]+[128,128,128]+[N_var**2], \
            "Sigma": [N_var]+[96,96]+[N_var**2]}

## Data loading parameters
# (supports loading of multiple files from the given dir. as long as they are numbered consecutively, 
#  nfile0 sets the first file to start loading from)
N_files = 1       # Number of files to load in a sequence to load longer trajectories
                  # (assuming trajectories are split into multiple files)

data_dir  = "data/" # base file-directory
file_name = "velocity_gradients_{:03d}.bin" # file-namespace
nfile0    = 1     # ID of the first file to load (useful for skipping a transient period in a sequence of output files)

N_part_per_file    = 1000 # Number of particles saved in each reference data-file (reduced number of example data)
N_steps_per_file   = 500  # Number of timesteps saved in each reference data-file  
dt_data = 1e-3            # dt of reference data 
                          # (should be equal to or a clean fraction of dt above to avoid supersampling artifacts)



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

        self.dt = dt
        self.dt_data = dt_data
        self.N_steps = N_steps

        self.N_up = N_up
        self.N_un = N_un
        self.N_sbins = N_sbins
        self.N_iter_layer = N_iter_layer
        self.channels = channels

        self.transf_layer = transf_layer
        self.distribution = distribution

        self.lr = lr
        self.N_epochs = N_epochs
        self.G_network_file = G_network_file
        self.logf_network_file = logf_network_file
        self.out_dir = out_dir
        self.write_states_freq = write_states_freq

        self.data_dir = data_dir
        self.file_name = file_name
        self.nfile0 = nfile0
        

        self.N_part_file = min(world_size*N_part, N_part_per_file) #131072)
        self.N_part_tot      = N_part*world_size
        self.N_files = max(N_files,self.N_part_tot//self.N_part_file) # ensure enough particles are loaded in any case

    def train(self, task):
        ##################################################################
        ### INITIALIZATION
        ##################################################################

        init_parallel(self,task)
        init_variables(self)
        init_model(self,task)

        params = list(self.G_model.parameters()) #+ list(self.Sigma_model.parameters())
        self.optimizer = to.optim.Adam(params, lr=self.lr) 
        self.optimizer.zero_grad()

        if(self.rank==0):
            os.makedirs(self.out_dir+"/states/",exist_ok=True)
            os.makedirs(self.out_dir+"/stats/",exist_ok=True)

            # empty logfile
            with open(self.out_dir+"/loss.txt", "w") as f:
                f.write("Epoch\tTime\tLoss\tSpectral-Loss\tCond.-Loss\n")

        ##################################################################
        ### LOADING OF REFERENCE DATA
        ##################################################################
        A = to.zeros([self.N_files*self.N_steps_per_file, self.N_part, 3, 3], device=self.device).detach()
        for it in range(self.N_files):
            sys.stdout.flush()
            fn  = self.data_dir+self.file_name.format(self.nfile0+it)
            A[it*self.N_steps_per_file:(it+1)*self.N_steps_per_file] = \
                    IO.read_binary_DNS(fn, self.N_part*self.world_size, self.N_steps_per_file, self.rank, self.world_size)

        tau_eta = to.einsum("tsij,tsij->ts", A, A).mean().to(self.device)
        dist.all_reduce(tau_eta, op=dist.ReduceOp.SUM)
        tau_eta/=self.world_size
        tau_eta = to.sqrt(1/tau_eta).to("cpu")
        if self.rank==0:
            print("Kolmogorov time ", tau_eta.item())
            sys.stdout.flush()
        A = A*tau_eta
        A = A.detach()
        Nstride = round(self.dt/self.dt_data) 
        #
        s = A.shape
        A_DNS = A.reshape(s[0],s[1],9)[::Nstride,:self.N_part,:self.N_var]
        #
        del A
        self.batch_size = self.N_part//self.N_batches
        train_loader = to.utils.data.DataLoader(to.transpose(A_DNS, 1, 0), batch_size=self.batch_size)#split along samples

        ##################################################################
        ### TRAINING
        ##################################################################

        for epoch in range(self.N_epochs):
            #############################
            ## EPOCH OPTIMIZATION

            epoch_loss = 0
            epoch_loss_sp   = 0
            epoch_loss_der  = 0
            i0 = to.randint(max(1,self.N_files*self.N_steps_per_file//Nstride-self.N_steps),(1,))[0]
            
            time0 = time.time() 
            for batch_index, training_sample in enumerate(train_loader):
                A_targ   = to.transpose(training_sample, 1, 0)[i0:i0+self.N_steps].to(self.device)
                dA = (A_targ[2:]-A_targ[:-2]).reshape(-1,self.N_var)/(2*self.dt)
                
                B = A_targ[1:-1].reshape(-1,self.N_var)
                B.requires_grad = True
                N, _ = self.rhs(B)
                B.requires_grad = False
                aux = to_matrix_extended((dA-N).reshape(-1,1,self.N_var))

                T = to_matrix_extended(B.reshape(-1,1,self.N_var))
                norm2 = to.einsum("tsij,tsij->ts", T, T)
                S  = .5*(T + to.transpose(T, -1,-2))
                W  = T - S
                S2 = to.einsum("tsik,tskj->tsij", S, S)
                W2 = to.einsum("tsik,tskj->tsij", W, W)
                SW = to.einsum("tsik,tskj->tsij", S, W) + to.einsum("tsik,tskj->tsij", W, S)
                B3 = to.einsum("tsik,tskj->tsij", S, W2) + to.einsum("tsik,tskj->tsij", W2, S) - \
                     to.einsum("tsik,tskj->tsij", S2, W) -  to.einsum("tsik,tskj->tsij", W, S2)
                
                loss_der = to.zeros((1,), device=dA.device)
                loss_der = to.mean(to.einsum("tsij,tsij->ts", S, aux)**2)
                loss_der+= to.mean(to.einsum("tsij,tsij->ts", W, aux)**2)
                loss_der+= to.mean(to.einsum("tsij,tsij->ts", S2,aux)**2/norm2)
                loss_der+= to.mean(to.einsum("tsij,tsij->ts", W2-SW,aux)**2/norm2)
                loss_der+= to.mean(to.einsum("tsij,tsij->ts", B3,aux)**2/norm2**1.5)
                
                #trajectories
                loss_sp = to.zeros((1,), device=self.device)

                A_ens    = to.zeros([self.N_steps, self.batch_size, self.N_var], device=self.device)
                A_ens[0] = A_targ[0]
                for it in range(self.N_steps-1):
                    A = A_ens[it].clone()
                    if it==0:
                        A.requires_grad=True

                    N0, div = self.rhs(A)
                    A1 = A + self.dt*N0
                    N1, _ = self.rhs(A1)
                    
                    A_ens[it+1] = (A + .5*self.dt*(N0 + N1)).float()

                loss_sp = self.compute_loss(A_ens, A_targ, epoch, derivative=False)

                #loss for each minibatch
                loss = loss_sp + beta*loss_der
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.G_model.zero_grad(set_to_none=True)
                

                dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
                dist.reduce(loss_sp, 0, op=dist.ReduceOp.SUM)
                dist.reduce(loss_der, 0, op=dist.ReduceOp.SUM)
                if self.rank==0:
                    print("\t Batch {}\t{:.3e}\t{:.3e}\t{:.3e}".format(batch_index, (loss.item()/self.world_size), (loss_sp.item()/self.world_size), (loss_der.item()/self.world_size)))
                    sys.stdout.flush()

                epoch_loss  += loss.detach()
                epoch_loss_sp  += loss_sp.detach()
                epoch_loss_der += loss_der.detach()
            
            #############################
            ## OUTPUT AND LOGGING

            epoch_loss = epoch_loss.item()/(self.N_batches)
            epoch_loss_sp = epoch_loss_sp.item()/(self.N_batches)
            epoch_loss_der = epoch_loss_der.item()/(self.N_batches)
            
            # Statistics
            if epoch%self.write_states_freq==0:
                with to.no_grad():
                    A = to_matrix_extended(A_ens.detach())
                    S = .5*(A + to.transpose(A,-1,-2))
                    corr_S = correlations( S ).detach().cpu().numpy()
                    corr_W = correlations(A-S).detach().cpu().numpy()
                    
                    if self.rank==0:
                        pickle.dump([corr_S,corr_W], open((self.out_dir+"stats/correlations_%d" % epoch)+"_MOD.bin", "wb"))

                    A = to_matrix_extended(A_targ.detach())
                    S = .5*(A + to.transpose(A,-1,-2))
                    corr_S = correlations( S ).detach().cpu().numpy()
                    corr_W = correlations(A-S).detach().cpu().numpy()
                    
                    if self.rank==0:
                        pickle.dump([corr_S, corr_W], open((self.out_dir+"stats/correlations_%d" % epoch)+"_DNS.bin", "wb"))
                        pickle.dump([A_ens[:,:32].cpu().numpy(), A_targ[:,:32].cpu().numpy()], open((self.out_dir+"stats/traj_%d.bin" % epoch), "wb"))
        
            # Logging and Network State
            if self.rank==0:
                print("Epoch {} ({} s)\t".format(epoch, float(time.time()-time0)),"Losses: ", epoch_loss, "\t", epoch_loss_sp,"\t", epoch_loss_der)
                sys.stdout.flush()
                with open(self.out_dir+"/loss.txt", "a") as fh:
                    fh.write(("%d\t%.2f\t%.6f\t%.6f\t%.6f\n" % (epoch, float(time.time()-time0), epoch_loss, epoch_loss_sp, epoch_loss_der)))
                
                if epoch%self.write_states_freq==0:
                    to.save(self.G_model.module.state_dict(), self.out_dir+"states/G_state_epoch_{}.pt".format(epoch))

            del A_ens, A, A1, N0, N1

        return


    def rhs(self, A):
        _, log_prob = self.log_f.log_probability(A, self.transf_layer)
        dF = gradient(log_prob, A)

        G = self.G_model(A).reshape(-1, self.N_var, self.N_var)
        T = G - to.transpose(G, -1, -2)
        N = to.zeros(A.shape, device=A.device)
        for i in range(self.N_var):
            N[:,i] += divergence(T[:,i], A)
        n = N + to.einsum("sij,sj->si", T, dF)
        div = -to.einsum("si,si->s", N, dF)

        return n, div


    def compute_loss(self, AA, BB, epoch, derivative=False):
    
        if derivative:
            A = (AA[2:] - AA[:-2])/(2*self.dt)
            B = (BB[2:] - BB[:-2])/(2*self.dt)
        else:
            A = AA
            B = BB
        
        sA = A.shape #time ,sample, 8  comp
        Nk = sA[0]//3
        omg = to.linspace(0,Nk-1,Nk).detach().to(A.device)

        A = to_matrix_extended(to.fft.rfft(A, axis=0)[:Nk]/Nk)
        sp_A = (to.einsum("ksij,kspq->ksijpq", A, to.conj(A))).mean(axis=1)
        sp_A1 = to.einsum("kijij->k", sp_A)
        sp_A2 = to.einsum("kijji->k", sp_A)
        E_A   = 1*to.real(to.sum(sp_A1))
        
        B = to_matrix_extended(to.fft.rfft(B, axis=0)[:Nk]/Nk)
        sp_B = (to.einsum("ksij,kspq->ksijpq", B, to.conj(B))).mean(axis=1)
        sp_B1 = to.einsum("kijij->k", sp_B)
        sp_B2 = to.einsum("kijji->k", sp_B)
        E_B   = 1*to.real(to.sum(sp_B1))
        
        loss = to.sum(omg*to.abs(sp_A1/E_A - sp_B1/E_B)**2)
        loss+= to.sum(omg*to.abs(sp_A2/E_A - sp_B2/E_B)**2)
        
        with to.no_grad():
            spect_A = 1*to.real(sp_A.detach())
            spect_B = 1*to.real(sp_B.detach())
        
            dist.reduce(spect_A, 0, op=dist.ReduceOp.SUM)
            dist.reduce(spect_B, 0, op=dist.ReduceOp.SUM)
            spect_A/=self.world_size
            spect_B/=self.world_size
            if self.rank==0:
                pickle.dump([spect_A.cpu().numpy(), spect_B.cpu().numpy()], open(self.out_dir+"stats/spectra_%d.bin" % epoch, "wb"))
        
        return loss


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
