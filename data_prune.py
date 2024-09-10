import torch as to
import numpy as np

def distribute(N, rank, ws):
    chunk_sz = N//ws
    rem = N - chunk_sz*ws
    l0 = rank*chunk_sz
    if rank < rem:
        chunk_sz+=1
        l0+=rank
    else:
        l0+=rem
    l1 = l0+chunk_sz
    if rank == ws-1:
        assert l1 == N
    return l0,l1

def read_binary_DNS(path, N_part, N_times, rank, nranks_per_file):
#DNS details
    chunk_sz = 9

#distribute data among gpus
    rank_IO = rank % nranks_per_file
    l0,l1 = distribute(N_part, rank_IO, nranks_per_file)
    N_part = l1-l0
    #if rank==0:
 #   print(rank, "reading binary", path,l0,l1)
#read the binary
    #L = chunk_sz*N_part*N_times
    #A = array.array('d')
    f = open(path, 'rb')
    #off = lo*chunk_sz*N_times
    #data = np.memmap(filename, dtype='d', mode='r', offset=off, shape=L)
    #A.fromfile(f, L) #, os.path.getsize(fn) // var.itemsize)

    off =   8*l0*N_times*chunk_sz #in bytes
    L   = N_part*N_times*chunk_sz #in dtype
    A   = np.memmap(f, dtype=np.float64, mode='r', offset=off, shape=L) #, order='C')

    return A

    A = to.tensor(A)
    #A = to.reshape(A, (N_part,N_times, 3,3))
    #A = A[l0:l1]
    #local number of particles
    #N_part = l1-l0
    A = to.reshape(A, (N_part,N_times, 3,3))
    #fortran ordering
    #A = to.reshape(A, (N_part,N_times, 3,3))
    #here we need C and we opted for layout: time,samples,components
    A = to.einsum("stij->tsji", A)

    return A
N_part=262144
N_times=500
N_part_cut=1000

A=read_binary_DNS("./data/tracked_part1_A_starting_034.bin",N_part,N_times,0,1)
print(A.shape)
A=A.reshape((N_part,N_times,9))
A_cut=A[:N_part_cut].reshape(-1)

A_cut.tofile("./data/tracked_part1_A_starting_034_cut.bin") # save to file

# double check
A_load=read_binary_DNS("./data/tracked_part1_A_starting_034_cut.bin",N_part_cut,N_times,0,1)
A_load=A_load.reshape((N_part_cut,N_times,9))

print(A.shape,A_load.shape)
print(np.abs(A_load-A[:N_part_cut]).sum())
print(A[:3,:3],A_load[:3,:3])
