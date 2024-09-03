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

def save_model(model,file):
    
    N_layers = len(model.s_func)
    dict_layers = dict()
    for k in range(N_layers):
        dict_layers["s_func_{}".format(k)] = model.s_func[k].module.state_dict()
        dict_layers["t_func_{}".format(k)] = model.t_func[k].module.state_dict()

    to.save(dict_layers, file)

    return

def load_model(fn, model, N_layers, device):
    checkpoint = to.load(fn, map_location=device)
    for k in range(N_layers):
        model.s_func[k].module.load_state_dict(checkpoint["s_func_{}".format(k)])
        model.t_func[k].module.load_state_dict(checkpoint["t_func_{}".format(k)])
    return model
