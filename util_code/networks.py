import torch as to
import torch.nn as nn
import torch.nn.functional as F
from .basic_util import to_matrix
from .rational_splines import unconstrained_rational_quadratic_spline, rational_quadratic_spline
to.set_default_dtype(to.float32)

class MLP(nn.Module):

    def __init__(self, channels):
        super().__init__()
        #
        N_layers = len(channels)
        activ  = nn.Softplus()
        #
        transf = nn.Linear(channels[0], channels[1])
        with to.no_grad():
            transf.weight*=1e-4
            transf.bias*=1e-4
        layer_list = [transf]
        for i in range(1,N_layers-1):
            transf = nn.Linear(channels[i], channels[i+1])
            with to.no_grad():
                transf.weight*=1e-4
                transf.bias*=1e-4
            layer_list.append(activ)
            layer_list.append(transf)
        #
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


class RealNVP(nn.Module):
    
    def __init__(self, masks, channels, which, distribution, task, ngpus):
        super(RealNVP, self).__init__()

        self.dim = len(masks[0])
        self.device = masks.device
        self.channels = channels

        #custom log distribution parametrized through the invariasnts
        self.distribution = distribution
        #mask the variables not updated across a layer
        self.masks = nn.ParameterList([nn.Parameter(mask, requires_grad=False) for mask in masks]).to(self.device)
        #original specification of layers
        self.N_layers = len(self.masks)
        self.TOLL = to.tensor(4., device=self.device).detach()
        #
        self.s_func = nn.ModuleList()
        for k,mask in enumerate(self.masks):
            #layers.append( RealNVPNode(mask, channels, which, k, N, task, ngpus).to(self.device) )
            self.s_func.append( nn.parallel.DistributedDataParallel(self.make_MLP_sequential(self.channels["aff"]).to(self.device), \
                        device_ids=[task] if ngpus>0 else None, find_unused_parameters=False) )
        self.s_func = self.s_func.to(self.device)
        #
        self.t_func = nn.ModuleList()
        for k,mask in enumerate(self.masks):
            #layers.append( RealNVPNode(mask, channels, which, k, N, task, ngpus).to(self.device) )
            self.t_func.append( nn.parallel.DistributedDataParallel(self.make_MLP_sequential(self.channels["aff"]).to(self.device), \
                           device_ids=[task] if ngpus>0 else None, find_unused_parameters=False) )
        self.t_func = self.t_func.to(self.device)
        
        return

    def log_probability(self, x, which):
        
        y = x.clone()
        log_prob = to.zeros(x.shape[0], device=self.device)
        for k in reversed(range(self.N_layers)):

            mask = self.masks[k]
            self.idx_up = mask.nonzero()[:,0]
            self.idx_un = (1.-mask).nonzero()[:,0]
            x_un = to.index_select(y, 1, self.idx_un)
            x_up = to.index_select(y, 1, self.idx_up)
            #
            s = self.TOLL*to.tanh(self.s_func[k](x_un).double()/self.TOLL)
            t = self.t_func[k](x_un)
            #
            y[:,self.idx_up] = (x_up - t)*to.exp(-s).float()
            log_prob -= s.sum(-1).float()
        #
        log_prob += self.distribution(y)

        return y, log_prob

    def rsample(self, x, which):
        
        y = x.clone()
        log_prob = self.distribution(y)
        for k in range(self.N_layers):

            mask = self.masks[k]
            self.idx_up = mask.nonzero()[:,0]
            self.idx_un = (1.-mask).nonzero()[:,0]
            x_un = to.index_select(y, 1, self.idx_un)
            x_up = to.index_select(y, 1, self.idx_up)
            #
            s = self.TOLL*to.tanh(self.s_func[k](x_un).double()/self.TOLL)
            t = self.t_func[k](x_un)
            #
            y[:,self.idx_up] = x_up*to.exp(s).float() + t
            log_prob += s.sum(-1).float()

        return y, log_prob

    def make_MLP_sequential(self, channels):

        N_layers = len(channels)
        activ  = nn.Softplus()
        transf = nn.Linear(channels[0], channels[1])
        with to.no_grad():
            transf.weight*=1e-4
            transf.bias*=1e-4

        layer_list = [transf]
        for i in range(1,N_layers-1):
            transf = nn.Linear(channels[i], channels[i+1])
            with to.no_grad():
                transf.weight*=1e-4
                transf.bias*=1e-4
            layer_list.append(activ)
            layer_list.append(transf)

        return nn.Sequential(*layer_list)


