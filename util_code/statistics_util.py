import numpy as np
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
import pickle
import sys
from .basic_util import to_matrix, gradient, divergence


figsz = 8
fontsz = 18
mpl.rcParams["lines.linewidth"] = 4

def conditional_moments(data, dt, batch_size, log_f, G_model, Sigma_model, self):

        N_bins  = 256
        min_inv = [  0,   0, -10, -10,   0]
        #invariants for conditioning
        N_invs = 5
        max_inv = [ 20,  20,  10,  10,  20]

        #pdf of the invariants
        pdf     = to.zeros([N_invs, N_bins],    device=self.device)
        #averages of A*A and A*A.T
        avg_da  = to.zeros([N_invs, N_bins, 2], device=self.device)
        avg_dg  = to.zeros([N_invs, N_bins, 2], device=self.device)
        avg_da2 = to.zeros([N_invs, N_bins, 2], device=self.device)
        avg_dg2 = to.zeros([N_invs, N_bins, 2], device=self.device)
        
        train_loader = to.utils.data.DataLoader(data, batch_size=batch_size)

        for batch_index, training_sample in enumerate(train_loader):

#network config [batch,comp]
            A  = training_sample[:,:self.N_var].to(self.device)
            dA = training_sample[:,self.N_var:].to(self.device)

            A.requires_grad = True
            _, log_prob = self.log_f.log_probability(A, self.transf_layer)
            dF = gradient(log_prob, A)
                ###f = to.exp(log_prob.double()).float()
#drift gauge and noise variance
            G = self.G_model(A).reshape(-1, self.N_var, self.N_var)
                ###v = v_model(A)
            T = G - to.transpose(G, -1, -2) #+ D   #eye(8)
#Gauge term * f has zero divergene
            N = to.einsum("sij,sj->si", T, dF)
            for i in range(self.N_var):
                N[:,i] += divergence(T[:,i], A)
#########
            V = dA - N*self.dt
            #if self.rank==0:
            #    print("Loss:", epoch_loss)
            #    sys.stdout.flush()
#########
            with to.no_grad():

                a    = to_matrix(A)
                invs = get_invariants(a)
                da = to_matrix(dA)
                dg = N*dt ###+ sqrt(2*dt)*to.einsum("sij,sj->si", Sigma, to.normal(A.shape, device=A.device))
                dg = to_matrix(dg)

                for l in range(N_invs):

                    inv = invs[l]
                    bins = to.linspace(min_inv[l], max_inv[l], N_bins+1, device=self.device)
                    bins = .5*(bins[1:]+bins[:-1])
                    dinv = bins[1]-bins[0]
                    for k in range(N_bins):
                        idx      = to.abs(inv-bins[k]) < dinv
                        pdf[l,k] = da[idx,:,:].size(dim=0)

                        s = .5*(a[idx,:,:] + to.transpose(a[idx,:,:],-1,-2))
                        w = a[idx,:,:]-s
                        avg_da[l,k,0] += to.einsum("sij,sij->", s, da[idx,:,:])
                        avg_da[l,k,1] += to.einsum("sij,sij->", w, da[idx,:,:])
                        avg_dg[l,k,0] += to.einsum("sij,sij->", s, dg[idx,:,:])
                        avg_dg[l,k,1] += to.einsum("sij,sij->", w, dg[idx,:,:])

                    for k in range(N_bins):
                        idx          = to.abs(inv-bins[k]) < dinv
                        avg_da2[l,k,0] += to.einsum("sij,sij->", da[idx,:], da[idx,:])
                        avg_da2[l,k,1] += to.einsum("sij,sji->", da[idx,:], da[idx,:])
                        avg_dg2[l,k,0] += to.einsum("sij,sij->", dg[idx,:], dg[idx,:])
                        avg_dg2[l,k,1] += to.einsum("sij,sji->", dg[idx,:], dg[idx,:])

        dist.reduce(pdf,     0, op=dist.ReduceOp.SUM)
        dist.reduce(avg_da,  0, op=dist.ReduceOp.SUM)
        dist.reduce(avg_dg,  0, op=dist.ReduceOp.SUM)
        dist.reduce(avg_da2, 0, op=dist.ReduceOp.SUM)
        dist.reduce(avg_dg2, 0, op=dist.ReduceOp.SUM)

        if self.rank==0:

            avg_da /=pdf[:,:,None]
            avg_dg /=pdf[:,:,None]
            avg_da2/=pdf[:,:,None]
            avg_dg2/=pdf[:,:,None]

            for l in range(N_invs):

                bins = to.linspace(min_inv[l], max_inv[l], N_bins+1, device=self.device)
                bins = .5*(bins[1:]+bins[:-1])

                #store
                buff = np.zeros((N_bins,1+2))
                buff[:,0]  = bins.to("cpu").numpy()
                buff[:,1:] = avg_da[l].to("cpu").numpy()
                with open(("avg_da_cond_inv_%d.txt" % (l+1)), "w") as f:
                    np.savetxt(f, buff)
                buff[:,1:] = avg_dg[l].to("cpu").numpy()
                with open(("avg_dg_cond_inv_%d.txt" % (l+1)), "w") as f:
                    np.savetxt(f, buff)

                #store2
                buff = np.zeros((N_bins,1+2))
                buff[:,0]  = bins.to("cpu").numpy()
                buff[:,1:] = avg_da2[l].to("cpu").numpy()
                with open(("avg_da2_cond_inv_%d.txt" % (l+1)), "w") as f:
                    np.savetxt(f, buff)
                buff[:,1:] = avg_dg2[l].to("cpu").numpy()
                with open(("avg_dg2_cond_inv_%d.txt" % (l+1)), "w") as f:
                    np.savetxt(f, buff)

                #store counter
                buff = np.zeros((N_bins,1+1))
                buff[:,0] = bins.to("cpu").numpy()
                buff[:,1] =  pdf[l].to("cpu").numpy()
                with open(("pdf_inv_%d.txt" % (l+1)), "w") as f:
                    np.savetxt(f, buff)

        return

def get_invariants(A):

    invs = to.zeros([5, A.shape[0]], device=A.device)
    
    S = 5e-1*(A + to.transpose(A, -1, -2))
    W = A - S
    S2 = to.matmul(S, S)
    W2 = to.matmul(W, W)
    SW2= to.matmul(S, W2)

#invariants
    invs[0] = to.einsum("skk->s", S2)
    invs[1] =-to.einsum("skk->s", W2)
    invs[2] = to.einsum("skk->s", to.matmul(S, S2))
    invs[3] = to.einsum("skk->s", SW2)
    invs[4] =-to.einsum("skk->s", to.matmul(S, SW2))

    return invs


def correlations(S):

        #sample, ij, time layout
        with to.no_grad():
            N_times = S.shape[0]
            S = to.einsum("tsij->sijt", S).detach()
            aux    = 0j
            aux    = to.fft.rfft(S)
            S2_hat = to.real(to.einsum("sijk,sijk->sk", aux, to.conj(aux)))
            S2_hat = to.mean(S2_hat, axis=0)

            corr_S = to.fft.irfft(S2_hat)[:N_times//2].detach()
            dist.reduce(corr_S, 0, op=dist.ReduceOp.SUM)
            #if self.rank == 0:
            #    corr_S/= corr_S[0].clone()
            #
        return corr_S

def plot_density_from_data(model, data, gen, num_bins, bounds, true_dist=None, log_VE=0, rank=0, ws=1, my_layer=-1):

    device = data.device
    N_samples = data.shape[0]

############################################RQ pdf
    bounds = [[-6,6], [-6,6]]
    print("RQ pdf todo",rank,ws,my_layer)
    RQ_pdf,bins1, bins2 = joint_pdf(data, "RQ", 392, bounds, rank, ws)
    RQ_pdf_mod, _, _    = joint_pdf(gen,  "RQ", 392, bounds, rank, ws)
    print("RQ pdf done")
    sys.stdout.flush()

    dist.barrier()
    if my_layer>=0:
        if rank==0:
            RQ_pdf_mod = np.maximum(RQ_pdf_mod, 1e-10)
            pickle.dump([RQ_pdf_mod,bins1,bins2], open(("pdf_RQ_MOD_layer_%d.bin" % my_layer), "wb"))
        return

    if rank==0:

        plt.rc("text.latex", preamble=r"\usepackage{bm}")

        RQ_pdf     = np.maximum(RQ_pdf,     1e-10)
        RQ_pdf_mod = np.maximum(RQ_pdf_mod, 1e-10)
        pickle.dump([RQ_pdf,bins1,bins2],     open("pdf_RQ_DNS.bin", "wb"))
        pickle.dump([RQ_pdf_mod,bins1,bins2], open("pdf_RQ_MOD.bin", "wb"))


        #font = {'family' : 'normal',
        #'weight' : 'bold',
        font = {'size'   : 16}
        mpl.rc('font', **font)

        mcmap = plt.cm.get_cmap("jet").copy()
        #mcmap.set_bad("white", .2)

        vmin=-4; vmax=1;
        lev = np.linspace(vmin,vmax,vmax-vmin+1)
        vmin=10**vmin; vmax=10**vmax;
        lev = 10**lev

        im_kwargs = {\
        "cmap": mcmap,\
        "extent": (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]), \
        "vmin":vmin, "vmax":vmax}

        fig, ax = plt.subplots(1,1,figsize=[figsz,figsz])
        cs = ax.contour(bins1, bins2, RQ_pdf.T, levels = lev, cmap=mcmap, norm=mpl.colors.LogNorm(), linewidths=4)
        im = ax.imshow(RQ_pdf.T, alpha=.6, **im_kwargs, origin="lower", norm=mpl.colors.LogNorm(), interpolation="bilinear")

        cs = ax.contour(bins1, bins2, RQ_pdf_mod.T, levels = lev, colors="black", norm=mpl.colors.LogNorm(), linewidths=2, linestyles="--")
        #cs = ax.contour(bins1, bins2, RQ_pdf_mod, levels = lev, cmap=mcmap, norm=mpl.colors.LogNorm(), linewidths=1, linestyles="--")

        ax.set_xlabel(r"$-\rm{Tr}\left(\mathbf{A}^3\right)/3$", fontsize=fontsz)
        ax.set_ylabel(r"$-\rm{Tr}\left(\mathbf{A}^2\right)/2$", fontsize=fontsz)
        ax.set_xticks([-4,-2,0,2,4])
        ax.set_yticks([-4,-2,0,2,4])

        sz = .93
        off = .0
        ax.set_position([0,0,sz,sz])
        cb_ax = fig.add_axes([1.05*sz,off,.04,sz-off]) 
        cb = fig.colorbar(im, cax=cb_ax, pad=.02)
        #cb.ax.get_yaxis().labelpad = 18
        #cb.ax.set_ylabel(r"PDF", rotation=270)

        plt.show()
        fig.savefig("pdf_RQ.pdf", format="pdf", bbox_inches='tight')


############################################S inv pdf
    bounds = [[0,8],[-8,8]]
    S_pdf,bins1, bins2 = joint_pdf(data, "S", 392, bounds, rank, ws)
    S_pdf_mod, _, _    = joint_pdf(gen,  "S", 392, bounds, rank, ws)

    if rank==0:
        
        pickle.dump([S_pdf,    bins1,bins2], open("pdf_S_DNS.bin", "wb"))
        pickle.dump([S_pdf_mod,bins1,bins2], open("pdf_S_MOD.bin", "wb"))
    
        mcmap = plt.cm.get_cmap("jet").copy()
        #mcmap.set_bad("white", .2)
        
        vmin=-4; vmax=1;
        lev = np.linspace(vmin,vmax,vmax-vmin+1)
        vmin=10**vmin; vmax=10**vmax;
        lev = 10**lev

        im_kwargs = {\
        "cmap": mcmap,\
        "extent": (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]), \
        "vmin":vmin, "vmax":vmax}

        fig, ax = plt.subplots(1,1,figsize=[figsz,figsz])
        cs = ax.contour(bins1, bins2, S_pdf.T, levels = lev, cmap=mcmap, norm=mpl.colors.LogNorm(), linewidths=4)
        im = ax.imshow(S_pdf.T, alpha=.6, **im_kwargs, origin="lower", norm=mpl.colors.LogNorm(), interpolation="bilinear")

        cs = ax.contour(bins1, bins2, S_pdf_mod.T, levels = lev, colors="black", norm=mpl.colors.LogNorm(), linewidths=2, linestyles="--")
        #cs = ax.contour(bins1, bins2, S_pdf_mod, levels = lev, cmap=mcmap, norm=mpl.colors.LogNorm(), linewidths=1, linestyles="--")
        #ax.set_aspect("equal")

        ax.set_xlabel(r"$\rm{Tr}\left(\mathbf{S}^2\right)$", fontsize=fontsz)
        ax.set_ylabel(r"$\rm{Tr}\left(\mathbf{S}^3\right)$", fontsize=fontsz)
        #ax.set_xticks([0,2,4,6,8,10])
        #ax.set_yticks([-4,-2,0,2,4])
        ax.set_ylim([-7,3])
        sz = .93
        off = .0
        ax.set_position([0,0,sz,sz])
        cb_ax = fig.add_axes([1.05*sz,off,.04,sz-off]) 
        cb = fig.colorbar(im, cax=cb_ax, pad=.02)
        #cb = fig.colorbar(im, ax=ax, pad=.02)
        #cb.ax.get_yaxis().labelpad = 18
        #cb.ax.set_ylabel(r"PDF", rotation=270)

        plt.show()
        fig.savefig("pdf_S.pdf", format="pdf", bbox_inches='tight')



############################################joint align pdf
    bounds = [[0,1],[0,1]]
    jalign_pdf,bins1, bins2 = joint_pdf(data, "jalign", 256, bounds, rank, ws)
    jalign_pdf_mod, _, _    = joint_pdf(gen,  "jalign", 256, bounds, rank, ws)
    if rank == 0:
        pickle.dump([jalign_pdf,    bins1,bins2], open("pdf_jalign_DNS.bin", "wb"))
        pickle.dump([jalign_pdf_mod,bins1,bins2], open("pdf_jalign_MOD.bin", "wb"))


################################################alignments pdf
    align_pdf, bins  = single_pdf(data, "align", 128, [0,1], rank, ws)
    align_pdf_mod, _ = single_pdf(gen,  "align", 128, [0,1], rank, ws)

    if rank==0:

        pickle.dump([align_pdf,    bins], open("pdf_align_DNS.bin", "wb"))
        pickle.dump([align_pdf_mod,bins], open("pdf_align_MOD.bin", "wb"))

        fig, ax = plt.subplots(1,1,figsize=[figsz,figsz])
        for i, pdf in enumerate(align_pdf):
            ax.plot(bins, pdf, label=(r"i=%d" % (i+1)))

        for i, pdf in enumerate(align_pdf_mod):
            ax.plot(bins, pdf, "--", color="black")

        ax.set_xlabel(r"$\left|\widehat{\omega}_i\right|$", fontsize=fontsz)
        ax.set_ylabel(r"PDF", fontsize=fontsz)
        #ax.set_xticks([0,2,4,6,8,10])
        #ax.set_yticks([-4,-2,0,2,4])

        plt.show()
        fig.savefig("pdf_align.pdf", format="pdf", bbox_inches='tight')



################################################Cart pdf
    pdf, bins  = single_pdf(data, "Cart", 1024, [-6,6], rank, ws)
    pdf_mod, _ = single_pdf(gen,  "Cart", 1024, [-6,6], rank, ws)
    if rank==0:
        pickle.dump([pdf,    bins], open("pdf_Cart_DNS.bin", "wb"))
        pickle.dump([pdf_mod,bins], open("pdf_Cart_MOD.bin", "wb"))

################################################magnitude pdf
    pdf, bins  = single_pdf(data, "mag", 1024, [0,24], rank, ws)
    pdf_mod, _ = single_pdf(gen,  "mag", 1024, [0,24], rank, ws)
    if rank==0:
        pickle.dump([pdf,    bins], open("pdf_mag_DNS.bin", "wb"))
        pickle.dump([pdf_mod,bins], open("pdf_mag_MOD.bin", "wb"))

    pdf, bins  = single_pdf(data, "log_mag", 1024, [-5,1.7], rank, ws)
    pdf_mod, _ = single_pdf(gen,  "log_mag", 1024, [-5,1.7], rank, ws)
    if rank==0:
        pickle.dump([pdf,    bins], open("pdf_log_mag_DNS.bin", "wb"))
        pickle.dump([pdf_mod,bins], open("pdf_log_mag_MOD.bin", "wb"))

    return



def single_pdf(data, which, num_bins, bounds, rank, ws):

    device = data.device
    N_samples = data.shape[0]

#pdf from the data
    A = to_matrix(data)

    pdfs = []
    if which == "align":
        S = .5*(A + to.transpose(A,-1,-2))
        W = A-S
        lam, V = np.linalg.eigh(S.detach().cpu())
        #lam = lam.cpu().detach().numpy()
        #V   =   V.cpu().detach().numpy()
        W   =   W.cpu().detach().numpy()
        W = np.matmul(np.matmul(np.transpose(V, (0,2,1)), W), V)
#vorticity components
        omg = 1*lam
        #reverse the order
        omg[:,0] = np.abs(W[:,0,1])
        omg[:,1] = np.abs(W[:,0,2])
        omg[:,2] = np.abs(W[:,1,2])
        omg2 = np.linalg.norm(omg, axis=-1)
        omg = omg
        x = [omg[:,0]/omg2, omg[:,1]/omg2, omg[:,2]/omg2]
    elif which == "Cart":
        A = A.cpu().detach().numpy()
        x = [ np.concatenate((A[:,0,0],A[:,1,1])), np.concatenate((A[:,0,1],A[:,0,2],A[:,1,2])) ]
    elif which == "mag":
        S = .5*(A + to.transpose(A,-1,-2))
        W = A-S
        S = S.cpu().detach().numpy()
        W = W.cpu().detach().numpy()
        x = [np.einsum("sij,sij->s", S, S), np.einsum("sij,sij->s", W, W)]
    elif which == "log_mag":
        S = .5*(A + to.transpose(A,-1,-2))
        W = A-S
        S = S.cpu().detach().numpy()
        W = W.cpu().detach().numpy()
        x = [np.log10(np.einsum("sij,sij->s", S, S)), np.log10(np.einsum("sij,sij->s", W, W))]

    for v in x:
        pdf, bins1 = np.histogram(v, bins=num_bins, range=bounds, density=False)
        bins1 = .5*(bins1[1:]+bins1[:-1])
        db1 = bins1[1]-bins1[0]
        pdf = to.tensor(pdf).to(device)
        dist.all_reduce(pdf, op=dist.ReduceOp.SUM)
        pdf = pdf.detach().cpu().numpy()/(N_samples*ws*db1)
        if(rank==0):
            pdfs.append(pdf)

    return pdfs, bins1



def joint_pdf(data, which, num_bins, bounds, rank, ws):

    device = data.device
    N_samples = data.shape[0]

#pdf from the data
    A = to_matrix(data)
    if which == "RQ":
        A2 = to.matmul(A, A)
        y = -1/2.*to.einsum("sii->s", A2).detach().cpu().numpy()
        x = -1/3.*to.einsum("sij,sji->s", A2, A).detach().cpu().numpy()
    elif which == "S":
        S = .5*(A + to.transpose(A,-1,-2))
        S2 = to.matmul(S, S)
        x = to.einsum("sii->s", S2).detach().cpu().numpy()
        y = to.einsum("sij,sji->s", S2, S).detach().cpu().numpy()
    elif which == "jalign":
        S = .5*(A + to.transpose(A,-1,-2))
        W = A-S
        _, V = np.linalg.eigh(S.detach().cpu())
        #lam = lam.cpu().detach().numpy()
        #V   =   V.cpu().detach().numpy()
        W =  W.cpu().detach().numpy()
        W = np.matmul(np.matmul(np.transpose(V, (0,2,1)), W), V)
#vorticity components
        omg = 1*W[:,0,:]
        #reverse the order
        omg[:,0] = np.abs(W[:,0,1])
        omg[:,1] = np.abs(W[:,0,2])
        omg[:,2] = np.abs(W[:,1,2])
        omg2 = np.sum(omg**2, axis=-1)
        x = omg[:,0]**2/omg2
        y = omg[:,1]**2/omg2
    pdf, bins1, bins2 = np.histogram2d(x, y, bins=[num_bins,num_bins], \
            range=bounds, density=False)

    bins1 = .5*(bins1[1:]+bins1[:-1])
    bins2 = .5*(bins2[1:]+bins2[:-1])
    db1 = bins1[1]-bins1[0]
    db2 = bins2[1]-bins2[0]

    pdf = to.tensor(pdf).to(device)
    dist.all_reduce(pdf, op=dist.ReduceOp.SUM)
    pdf = pdf.detach().cpu().numpy()/(N_samples*ws*db1*db2)

    return pdf, bins1, bins2


def plot_density(model, data, num_bins, bounds, true_dist=None, log_VE=0, rank=0, ws=1):
    #x_mesh, y_mesh = np.meshgrid(np.linspace(-mesh_size, mesh_size, num=num_samples),
    #                             np.linspace(- mesh_size, mesh_size, num=num_samples))

    #cords = np.stack((x_mesh, y_mesh), axis=2)
    #cords_reshape = cords.reshape([-1, 2])
    #log_prob = np.zeros((num_samples ** 2))
    #for i in range(0, num_samples ** 2, num_samples):
    #    data = torch.from_numpy(cords_reshape[i:i + num_samples, :]).float()
    #    log_prob[i:i + num_samples] = model.log_probability(data).cpu().detach().numpy()
    #log_prob = model.log_probability(to.tensor(cords_reshape).float()) 

    #if log_VE != 0:
    #    log_prob -= log_VE(torch.tensor(cords_reshape).float())
    #log_prob = log_prob.reshape(num_samples,num_samples).detach().numpy()

    device = data.device
    N_samples = data.shape[0]

    mcmap = plt.cm.get_cmap("jet").copy()
    mcmap.set_bad("white", .2)

    A = to_matrix(data)
    A2 = to.matmul(A, A)
    y = -1/2.*to.einsum("sii->s", A2).detach().cpu().numpy()
    x = -1/3.*to.einsum("sij,sji->s", A2, A).detach().cpu().numpy()
    prob, bins1, bins2 = np.histogram2d(x, y, bins=[num_bins,num_bins], \
            range=bounds, density=False)

    prob = to.tensor(prob).to(device)
    dist.reduce(prob, 0, op=dist.ReduceOp.SUM)

    bins1 = .5*(bins1[1:]+bins1[:-1])
    bins2 = .5*(bins2[1:]+bins2[:-1])
    db1 = bins1[1]-bins1[0]
    db2 = bins2[1]-bins2[0]
    x_mesh, y_mesh = np.meshgrid(bins1, bins2, indexing="ij")
    coords = np.stack((x_mesh, y_mesh), axis=-1)
    r   = lam_inv(to.tensor(x_mesh).flatten(), to.tensor(y_mesh).flatten())
    Lam = to.zeros([num_bins**2,3,3], device=device)
    for i in range(3):
        Lam[:,i,i] = r[:,i]
    Lam = Lam.reshape(-1,9)[:,:8].float()
    print(to.max(Lam), "max Lam")

    log_prob_mod = model.log_probability(Lam).to(device).float()
    log_prob_mod = log_prob_mod.detach().cpu().numpy()/np.log(10.)
    log_prob_mod = log_prob_mod.reshape(num_bins,num_bins)

    if rank==0:
        
        prob = prob.detach().cpu().numpy()/(N_samples*ws*db1*db2)
        #prob/= (10**(log_VE(to.tensor(coords.reshape(-1,2))).detach().numpy().reshape(num_bins,num_bins)/np.log(10.)))
        norm = np.sum(db1*db2*prob)
        prob/=norm
        log_prob = np.where(prob>0, np.log10(prob), np.NaN)

        vmin=-6; vmax=2;
        lev = np.linspace(vmin,vmax,vmax-vmin+1)

        fig, ax = plt.subplots()
        cs = ax.contour(bins1, bins2, log_prob.T, levels = lev, colors = "black", linewidths=2) #cmap="jet")
    #ax.imshow(cords_reshape[:, 0], cords_reshape[:, 1], c=log_prob) #c=np.exp(log_prob))
        if true_dist is not None:
            ax.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', alpha=.05)

#    mpl.colorbar.ColorbarBase(ax=ax)
        im_kwargs = {\
        "cmap": mcmap,\
        "extent": (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]), \
        "vmin":vmin, "vmax":vmax}
        im = ax.imshow(log_prob.T, alpha=.6, **im_kwargs, origin="lower") #, interpolation="bilinear", norm=mpl.colors.LogNorm())

        mask = coords[:,:,0]**3 - 6*coords[:,:,1]**2 > 0
        prob = 10**log_prob_mod
        prob = np.where(mask, 10**log_prob_mod, 0)
        norm = np.sum(db1*db2*prob)
        log_prob_mod -= np.log10(norm)
        print(np.min(log_prob_mod[mask]),np.max(log_prob_mod[mask]))
        
        #interp = LinearNDInterpolator(coords, log_prob)  # NearestNDInterpolator  #list(zip(bins1, bins2))
        #log_pdf_mesh = interp(x_mesh, y_mesh)
        cs = ax.contour(bins1, bins2, log_prob_mod.T, levels = lev, linewidths=1, cmap=mcmap, linestyles="--") #colors="black"
        #cs = ax.scatter(x,y, c=log_prob_mod.T, levels = lev, linewidths=1, cmap=mcmap, linestyles="--") #colors="black"

        plt.show()

        fig.savefig("pdf_x.pdf", format="pdf")

    return



def plot_each_step(model, num_samples=200):
    data = model.sample_each_step(num_samples)
    len_data = len(data)

    fig, axis = plt.subplots(2, int((len_data+1)/2), figsize=(15, 10),
                             sharex=True, sharey=True)
    p = inflect.engine()

    num_plot = 0
    for i in range(len_data):
        if i == round((len_data+1)/2):
            axis.flatten()[num_plot].axis('off')
            num_plot += 1

        d = data[i]
        ax = axis.flatten()[num_plot]
        if i == 0:
            title = 'Original data'
        else:
            title = p.ordinal(i) + ' layer'

        ax.scatter(d[:, 0], d[:, 1], alpha=.2)
        ax.set_title(title)
        num_plot += 1


def generate_image_mask(in_channels, image_width, num_layers):
    count = 0
    vec = []
    for i in range(image_width**2*in_channels):
        count += 1
        if i % image_width == 0 and image_width % 2 == 0:
            count += 1
        vec.append(count % 2.)
    mask = to.tensor(vec).reshape(in_channels, image_width, image_width)
    masks = []
    for i in range(num_layers):
        if i % 2 == 0:
            masks.append(mask)
        else:
            masks.append(1. - mask)
    return masks



def lam_inv(x1, x3):
    N_samples = x1.shape[0]
    device = x1.device

    eps1 = -.5 + .5*sqrt(3.)*1j
    eps2 = -.5 - .5*sqrt(3.)*1j
    
    p = -x1/2
    q = -x3/3
    delta = q**2/4. + p**3/27.
    delta = to.where(delta<=0., (-.5*q + 1j*to.sqrt(-delta))**(1./3.), 0*delta+0j)
    print("CHECK delta", to.max(to.abs(delta))) 
    r = to.zeros([N_samples, 3], device=device)
    C =    1*delta
    r[:,0] = to.where(to.abs(C)>0., to.real(C - p/(3*C)), to.real(0*C))
    C = eps1*delta
    r[:,1] = to.where(to.abs(C)>0., to.real(C - p/(3*C)), to.real(0*C))
    #r[:,1] = to.real(C - p/(3*C))
    C = eps2*delta
    r[:,2] = to.where(to.abs(C)>0., to.real(C - p/(3*C)), to.real(0*C))
    #r[:,2] = to.real(C - p/(3*C))
    print("CHECK r", to.max(to.abs(r))) 
    return r
