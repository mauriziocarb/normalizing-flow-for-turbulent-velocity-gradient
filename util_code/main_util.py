import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from itertools import combinations

from .networks import RealNVP,MLP
import util_code.IO as IO

to.set_default_dtype(to.float32)

def init_parallel(obj, task):
    #global rank of this task
    obj.rank = obj.node*obj.ntasks + task
    print('local task', task, "global", obj.rank, "starting on node", obj.node)

#set a single gpu for each task on cpu
    if obj.ngpus > 0:
        obj.device = "cuda:"+str(task) #eg cuda:1
        to.cuda.set_device(task)
        be = "nccl" #best backend with GPUs
        if obj.rank == 0:
            print("training on GPU")
    else:
        obj.device = "cpu"
        be = "gloo" #backend for CPUs only
        if obj.rank == 0:
            print("training on CPU")

    dist.init_process_group(backend=be, init_method='env://', world_size=obj.world_size, rank=obj.rank)

    seed=to.zeros((1,),dtype=to.int64,device=obj.device) 
    if obj.rank==0:
        seed=to.randint(to.iinfo(to.int64).max,(1,),device=obj.device)   
    dist.broadcast(seed,src=0)
    to.manual_seed(seed[0])
    dist.barrier()

    return

def init_variables(obj):
    obj.N_var = obj.N_up+obj.N_un
    a = range(obj.N_var)
    var_comb = set(combinations(a, obj.N_up))
    num_layers = len(var_comb)
    masks = to.zeros([obj.N_iter_layer*num_layers, obj.N_var]).float()
    k = 0
    for i in range(obj.N_iter_layer):
        for c in var_comb:
            masks[k,c] = 1.
            k+=1
    obj.masks = masks.to(obj.device)
    del masks
    
    return

def init_model(obj, task):
    #model specification
    model = RealNVP(obj.masks, obj.channels, obj.transf_layer, obj.distribution, task, obj.ngpus)

    #parallelize model through devices
    model = model.to(obj.device)
    #load previous state
    N_layers = len(model.s_func)
    if not obj.logf_network_file == None:
        model = IO.load_model(obj.logf_network_file, model, N_layers, obj.device)
    model.eval()
    obj.log_f = model
    obj.log_f.zero_grad()

    if "G" in obj.channels.keys():
        #parallelize model through devices
        model = MLP(obj.channels["G"]).to(obj.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[task] if obj.ngpus > 0 else None, find_unused_parameters=False)
        #load previous state
        if not obj.G_network_file == None:
            checkpoint = to.load(obj.G_network_file, map_location=obj.device)
            model.module.load_state_dict(checkpoint)
        model.eval()
        obj.G_model = model
        obj.G_model.zero_grad()
    return


        