import os
import argparse
import yaml
import math
import matplotlib.pyplot as plt
import scipy
import pickle
import numpy as np 
import torch
import torch.nn as nn
import torchdiffeq as tdeq
import torch.autograd as autograd
import time
import warnings
from argparse import Namespace
from scipy.stats import gaussian_kde
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpu = torch.cuda.device_count()
mult_gpu = False if num_gpu < 2 else True

def gen_data(datasize, dim):
    mean = 0
    variance = 1.0
    standard_normal_samples = torch.randn(datasize, dim)
    data = standard_normal_samples * variance + mean
    return data

def get_e_ls(out, num_e):
    e_ls = []
    for i in range(num_e):
        e_ls.append(torch.randn_like(out).to(device))  # returns standard normal
    return e_ls

def divergence_approx(out, x, e_ls=[], t = None, net = None):
    approx_tr_dzdx_ls = []
    Jac_norm_ls = []
    for e in e_ls:
        sigma0, d = 0.01, Xdim_flow
        if 'sigma0' in args_yaml['training']:
            sigma0 = args_yaml['training']['sigma0'] 
        sigma = sigma0 / torch.sqrt(torch.tensor(d)).float()
        out_e = net(x+sigma*e.float(),t)
        e_dzdx = (out_e - out)/sigma
        Jac_norm = torch.zeros(x.shape[0], 1).to(device)
        Jac_norm_ls.append(Jac_norm)
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx_ls.append(e_dzdx_e.view(x.shape[0], -1).sum(dim=1, keepdim=True))
    approx_tr_dzdx_out = torch.cat(approx_tr_dzdx_ls, dim=1).mean(dim=1)
    Jac_norm_out = torch.cat(Jac_norm_ls, dim=1).mean(dim=1)
    return approx_tr_dzdx_out, Jac_norm_out

def divergence_bf(dx, x):
    sum_diag = 0.
    for i in range(x.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(),
                                        x, create_graph=True)[0][:, i]
    return sum_diag.view(x.shape[0], 1)

class FCnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Xdim_flow = config.Xdim_flow
        hid_dims = tuple(map(int, config.hid_dims.split("-")))
        self.layer_dims_in = (self.Xdim_flow,) + hid_dims
        self.layer_dims_out = hid_dims + (self.Xdim_flow,)
        self.build_layers()

    def build_layers(self):
        self.layers = []
        for layer_in, layer_out in zip(self.layer_dims_in, self.layer_dims_out):
            self.layers.append(nn.Linear(layer_in, layer_out))
            if layer_out != self.Xdim_flow:
                self.layers.append(nn.Softplus(beta=20))
        self.layers = nn.Sequential(*self.layers)
            
    def forward(self, x, t):
        return self.layers(x)

class ODEFunc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.div_bf = False 

    def forward(self, t, x, velocity=False):
        if velocity:
            x.requires_grad_(True)
            out = self.model(x,t)
            return out
        def odefunc_wrapper(t, x):
            x, _, _ = x
            x = x.float()
            if self.logpx:
                if self.fix_e_ls:
                    if self.e_ls is None:
                        self.e_ls = get_e_ls(x, self.num_e)
                else:
                    self.e_ls = get_e_ls(x, self.num_e)
                if self.div_bf:
                    with torch.set_grad_enabled(True):
                        x.requires_grad_(True)
                        t.requires_grad_(True)
                        out = self.model(x,t)
                        divf = divergence_bf(out, x).to(device)
                        Jac_norm_out = torch.zeros_like(divf).to(device)
                else:
                    out = self.model(x,t)
                    divf, Jac_norm_out = divergence_approx(out, x, self.e_ls,
                                                        t = t, net = self.model)
            else:
                divf = torch.zeros(x.shape[0]).to(device)
                Jac_norm_out = torch.zeros_like(divf)
                out = self.model(x,t)
            return out, -divf, Jac_norm_out
        return odefunc_wrapper(t, x)

class CNF(nn.Module):
    def __init__(self, odefunc):
        super(CNF, self).__init__()
        self.odefunc = odefunc
    def forward(self, x, args, test=False, mult_gpu=False):
        self.odefunc.logpx = True
        integration_times = torch.linspace(
            args.Tk_1, args.Tk, args.num_int_pts+1).to(device)
        if test:
            self.odefunc.logpx = False  
        self.odefunc.num_e = args.num_e
        dlogpx = torch.zeros(x.shape[0]).to(device)
        dJacnorm = torch.zeros(x.shape[0]).to(device)
        self.odefunc.e_ls = None 
        self.odefunc.fix_e_ls = args.fix_e_ls
        self.odefunc.counter = 0
        
        out = self.odefunc(0,x,velocity=True)
        #out = self.odefunc(1,x,velocity=True)

        if args.use_NeuralODE is False:
            predz, dlogpx, dJacnorm = tdeq.odeint(
                self.odefunc, (x, dlogpx, dJacnorm), integration_times, method=args.int_mtd,
                rtol = args.rtol, atol = args.atol)
        else:
            predz, dlogpx, dJacnorm = tdeq.odeint_adjoint(
                self.odefunc, (x, dlogpx, dJacnorm), integration_times, method=args.int_mtd,
                rtol = args.rtol, atol = args.atol)
            
        if mult_gpu:
            return predz[-1], dlogpx[-1], dJacnorm[-1], out
        else:
            return predz, dlogpx, dJacnorm, out

def default_CNF_structure(config):
    model = FCnet(config).to(device)
    odefunc = ODEFunc(model).to(device)
    CNF_ = CNF(odefunc).to(device)
    return CNF_

def FlowNet_forward(xinput, CNF, ls_args_CNF, block_now, test = True, return_full = False):
    if block_now == 0:
        return xinput, 0
    else:
        ls_args_CNF = ls_args_CNF[:block_now]
        with torch.no_grad():
            predz_ls, dlogpx_ls = [], []
            for i, args_CNF in enumerate(ls_args_CNF):
                predz, dlogpx, _, _ = CNF(xinput, args_CNF, 
                                    test = test,
                                    mult_gpu = mult_gpu)
                if mult_gpu:
                    if i == 0:
                        predz_ls.append(xinput)
                        dlogpx_ls.append(torch.zeros(xinput.shape[0]).to(device))
                    predz_ls.append(predz)
                    dlogpx_ls.append(dlogpx)
                    xinput = predz
                else:
                    xinput = predz[-1]
                    if i == 0:
                        predz_ls.append(predz)
                        dlogpx_ls.append(dlogpx)
                    else:
                        predz_ls.append(predz[1:])
                        dlogpx_ls.append(dlogpx[1:])
        if mult_gpu is False:
            predz_ls = torch.cat(predz_ls, dim=0)
            dlogpx_ls = torch.cat(dlogpx_ls, dim=0)
        else:
            predz_ls = torch.stack(predz_ls, dim=0)
            dlogpx_ls = torch.stack(dlogpx_ls, dim=0)
        if return_full:
            return predz_ls, dlogpx_ls
        else:
            return predz_ls[-1], dlogpx_ls[-1]

def l2_norm_sqr(input, return_full = False):
    if len(input.size()) > 2:
        norms = 0.5*input.view(input.shape[0], -1).pow(2).sum(axis=1)
    else:
        norms = 0.5*input.pow(2).sum(axis=1)
    if return_full:
            return norms
    else:
        return norms.mean()

def gradient_E_target(input, k=20):
    diff = input.pow(2).sum(axis=1)
    return torch.log(1+torch.exp(-k*(torch.sqrt(diff)-c))).mean() + 0.5*(diff).mean()

# We recommend manually calculating the Jacobian for GMM, since the automatic differentiation sometimes causes numerical issues.
def jacobian_manual_GMM(input_tensor, means, variances):
    means = torch.tensor(means, device=device)
    variances = torch.tensor(variances, device=device)
    def target_distribution_GMM_manual(x, means, pi, variances):
        diff = x.unsqueeze(1) - means
        exp_term = torch.exp(-0.5 * torch.sum((diff ** 2) / variances, dim=-1))  # Shape: (batch_size, num_means)
        density = torch.sum(pi * exp_term / torch.sqrt(torch.prod(variances, dim=-1)), dim=-1) / (2 * torch.pi) ** (x.shape[1] / 2)  # Shape: (batch_size)
        return density
    pi = torch.tensor([1/num_means] * num_means, dtype=torch.float32).to(device)
    g = target_distribution_GMM_manual(input_tensor, means, pi, variances)
    diff = input_tensor.unsqueeze(1) - means
    exp_term = torch.exp(-0.5 * torch.sum((diff ** 2) / variances, dim=-1))
    dgd_x = torch.einsum('ij,ijk->ik', pi * exp_term / torch.sqrt(torch.prod(variances, dim=-1)), diff / variances) / (2 * torch.pi) ** (input_tensor.shape[1] / 2)  # Shape: (batch_size, dim)
    epsl = 1e-16
    dg_dx = dgd_x / (g.unsqueeze(-1) + epsl)  # Shape: (batch_size, dim)
    df_dx = beta * dg_dx + (1 - beta) * input_tensor
    return df_dx

def jacobian_funnel_dd(input_tensor, sigma = 0.5):
    d = input_tensor.shape[1]
    assert d >= 2, "The dimensionality of the input must be at least 2"
    z1 = input_tensor[:, 0]
    z2 = input_tensor[:, 1]
    z_rest = input_tensor[:, 1:]
    grad_z1 = (z1 / sigma**2) - (torch.sum(z_rest**2, dim=1) / (2 * torch.exp(z1))) + (d - 1) / 2
    grad_z1 = grad_z1.unsqueeze(1)
    grad_z_rest = z_rest / torch.exp(z1).unsqueeze(1)
    jacobian_vector = torch.cat([grad_z1,grad_z_rest], dim=1)
    return jacobian_vector

def JKO_loss_func(xinput, model, ls_args_CNF, beta):
    num_rk4 = len(ls_args_CNF)
    loss_div_tot, loss_Jac_tot = 0, 0
    xinput_ = xinput.clone()
    for k in range(num_rk4):
        args = ls_args_CNF[k]
        predz, dlogpx, lossJacnorm, v_field = model(xinput, args, test = False, mult_gpu = mult_gpu) 
        if mult_gpu:
            xpk = predz
            loss_div_tot += dlogpx.mean()
            loss_Jac_tot += args_training.lam_jac * lossJacnorm.mean()
        else:
            xpk = predz[-1]
            loss_div_tot += dlogpx[-1].mean()
            loss_Jac_tot += args_training.lam_jac * lossJacnorm[-1].mean()
        xinput = xpk
    if args_yaml['CNF']['dynamic']:
        raw_movement = 0
        for a, b in zip(predz[1:], predz[:-1]):
            raw_movement += ((a-b)**2).mean()
    else:
        raw_movement = l2_norm_sqr(xpk - xinput_)
    loss_W2_tot = raw_movement/delta_tk

    if torch.any(torch.isnan(xpk)) or torch.any(torch.isinf(xpk)):
        print('xpk:', xpk)
        raise ValueError("NaN or Inf values found in gradients")
    
    if Type == 'GMM_sphere':
        angles = np.linspace(0, 2 * np.pi, num_means, endpoint=False)
        if Xdim_flow == 2:
            means = [(c * np.cos(angle), c * np.sin(angle)) for angle in angles]
        else:
            means = [(c * np.cos(angle), c * np.sin(angle)) + (c,) * (Xdim_flow - 2) for angle in angles]
        variances = [[1] * Xdim_flow for i in range(len(means))]
        loss_V_dot = jacobian_manual_GMM(xpk, means=means, variances=variances)
        loss_V_dot = torch.sum(loss_V_dot * v_field, dim=1).mean()
    elif Type == 'GMM_cube':
        if Xdim_flow <= 5:
            means = np.array([[(c if bit == '1' else -c) for bit in format(i, f'0{Xdim_flow}b')] for i in range(num_means)])
        else:
            means = np.array([[(c if bit == '1' else -c) for bit in format(i, f'05b')] for i in range(num_means)])
            means = np.hstack([means, np.full((num_means, Xdim_flow - 5), c)])
        variances = [[1] * Xdim_flow for i in range(len(means))]
        loss_V_dot = jacobian_manual_GMM(xpk, means=means, variances=variances)
        loss_V_dot = torch.sum(loss_V_dot * v_field, dim=1).mean()
    elif Type == 'truncated':
        diff = xpk.pow(2).sum(axis=1)
        loss_V_dot = torch.log(1+torch.exp(-punishment*(torch.sqrt(diff)-c))).mean() + 0.5*(diff).mean()
    elif Type == 'exponential':
        if Xdim_flow <= 10:
            loss_V_dot = -torch.sign(xpk) * c * beta + xpk
        else:
            loss_V_dot = torch.zeros_like(xpk)
            loss_V_dot[:, :10] = -torch.sign(xpk[:, :10]) * c * beta + xpk[:, :10]
            loss_V_dot[:, 10:] = xpk[:, 10:]
        loss_V_dot = torch.sum(loss_V_dot * v_field, dim=1).mean()
    elif Type == 'funnel':
        loss_V_dot = jacobian_funnel_dd(xpk, sigma = 0.9)
        loss_V_dot = torch.sum(loss_V_dot * v_field, dim=1).mean()
    return loss_V_dot, loss_div_tot, loss_W2_tot, loss_Jac_tot, raw_movement

# Moves data through all given block (forward/backward), and return the intermediate results
def move_over_blocks(self, move_configs, nte = 1000):
    with torch.no_grad():
        Xtest = self.X_test.to(device)
        if move_configs.block_id > 1:
            Zhat_ls_prev, Zout = [Xtest], Xtest
            for self_mod in move_configs.self_ls_prev:
                Zout, _ = FlowNet_forward(Zout, self_mod.CNF, self_mod.ls_args_CNF, #ls_args_CNF consists of all sub-int
                                          self_mod.block_now, return_full = False)
                Zhat_ls_prev.append(Zout)
            Xtest = Zout
        # Begin current block:
        Zhat_ls, _ = FlowNet_forward(Xtest, self.CNF, self.ls_args_CNF,
                                     self.block_now, return_full = True)
        if mult_gpu:
            ids = range(len(Zhat_ls))
        else:
            ids = torch.linspace(0, Zhat_ls.shape[0]-1, self.block_now+1).long()
        Zhat_ls = [Zhat_ls[i] for i in ids]
        if move_configs.block_id > 1:
            Zhat_ls = Zhat_ls_prev + Zhat_ls[1:]
    return Zhat_ls 

def push_samples_forward(data_loader, self):
    X = []
    for xsample in data_loader:
        xsample = xsample[0]
        xpushed, _ = FlowNet_forward(xsample.to(device), self.CNF, 
                                    self.ls_args_CNF, self.block_now, test = True, return_full = False)
        X.append(xpushed)
    X = torch.cat(X, dim=0)
    return X

def on_off(self, on = True):
    for a in self.ls_args_CNF:
        a.int_mtd = 'dopri5' if on else 'rk4'

def load_prev_CNFs(load_configs):
    self_ls_prev = [] 
    for b in range(1, load_configs.block_id):
        self_prev = Namespace()
        filepath = os.path.join(load_configs.master_dir, f'{load_configs.prefix}{b}.pth')
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self_prev.CNF = default_CNF_structure(config = load_configs.vfield_config)
        self_prev.CNF.load_state_dict(checkpoint['model'])  # In .pth file
        self_prev.ls_args_CNF = checkpoint['ls_args_CNF']
        on_off(self_prev, on = True)
        self_prev.block_now = len(self_prev.ls_args_CNF)
        self_ls_prev.append(self_prev)
    return self_ls_prev

def loop_data_loader(dataloader):
    data_iterator = iter(dataloader)
    while True:
        try:
            yield next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)

def add_diffuse(x):
    eps = 1e-3 
    t, dt = eps, eps
    beta_min, beta_max = 0.1, 20
    beta_t = beta_min + t*(beta_max - beta_min)
    dt_term = -0.5*beta_t*x*dt
    dw = np.sqrt(dt)*torch.randn_like(x)
    dw_term = np.sqrt(beta_t)*dw
    dx = dt_term + dw_term
    return x + dx

def pdist(sample_1, sample_2, norm=2):
    return torch.cdist(sample_1, sample_2, p=norm)

class MMDStatistic:
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = 1. / (n_1 * n_1)
        self.a11 = 1. / (n_2 * n_2)
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)
        mmd_dict = {}
        for alpha in alphas:
            kernels = torch.exp(- alpha * distances**2)
            k_1 = kernels[:self.n_1, :self.n_1]
            k_2 = kernels[self.n_1:, self.n_1:]
            k_12 = kernels[:self.n_1, self.n_1:]
            mmd = self.a00 * k_1.sum() + self.a11 * k_2.sum() + 2 * self.a01 * k_12.sum()
            mmd_dict[alpha.item()] = f'{mmd.item():.2e}' 
        if ret_matrix:
            return mmd_dict, kernels
        else:
            return mmd_dict

def get_MMD(X, Xhat, nmax = 1000, alpha_ls = [0.5]):
    X, Xhat = torch.from_numpy(X).to(device), torch.from_numpy(Xhat).to(device)
    nmax1 = min(nmax, X.shape[0])
    nmax2 = min(nmax, Xhat.shape[0])
    X = X[torch.randperm(X.shape[0])[:nmax1]]
    Xhat = Xhat[torch.randperm(Xhat.shape[0])[:nmax2]]
    print(X.shape, Xhat.shape)
    distances = pdist(X,X)
    dist_median = torch.median(distances)
    gamma = 0.1*dist_median
    alpha_ls = [0.5/gamma**2]
    alpha_ls = torch.tensor(alpha_ls).to(device)
    mtd = MMDStatistic(nmax1, nmax2)
    mmd_dict = mtd(X, Xhat, alpha_ls)
    return mmd_dict

def helper(on = True):
    if on:
        self.CNF.module.odefunc.div_bf = True
        if block_id > 1:
            for self_mod in self_ls_prev:
                on_off(self_mod, on = False)
                self_mod.CNF.odefunc.div_bf = True
    else:
        self.CNF.module.odefunc.div_bf = False
        if block_id > 1:
            for self_mod in self_ls_prev:
                on_off(self_mod, on = True)
                self_mod.CNF.odefunc.div_bf = False


def plot_traj(Xtraj, d):
    ncol = len(Xtraj)
    fig, axs = plt.subplots(1, ncol, figsize=(4 * ncol, 4))
    fsize = 18
    ts = np.linspace(0, 1, ncol)
    for i, ax in enumerate(axs.flatten()):
        if d == 1:
            ax.hist(Xtraj[i].clone().detach().cpu().numpy(), bins=50, density=True)
        else:
            Xtmp = Xtraj[i].clone().detach().cpu().numpy()
            ax.scatter(Xtmp[:, 0], Xtmp[:, 1], s=0.5)
            ax.set_xlim([-15, 15])
            ax.set_ylim([-15, 15])
        if i < ncol - 3:
            ax.set_title(f'Block {i}', fontsize=fsize)
        else:
            ax.set_title(f'Int{i - (ncol - 3) + 1}', fontsize=fsize)
    fig.tight_layout()
    plt.savefig(os.path.join(directory, f'trajectory_{index}.png'))
    plt.close()

def get_beta(block_id, number=8):
    if block_id <= number:
        beta = block_id/number
    else:
        beta = 1
    return beta

def get_c_and_punishment(block_id):
    # Please increase the values accordingly if you set c > 8.
    if Type != 'truncated':
        warnings.warn("This function is designed for 'truncated' type only.")
        return None, None
    if Xdim_flow <= 5:
        if block_id <= 4:
            return min(4 , c1+0.1), 20
        elif 4 < block_id <= 5:
            return min(5, c1+0.1), 25
        elif 5 < block_id <= 6:
            return min(6, c1+0.1), 30
        elif 6 < block_id <= 7:
            return min(7, c1+0.1), 30
        elif block_id >= 8:
            return min(8, c1+0.1), 30
    else:
        if block_id <= 4:
            return min(4 , c1+0.1), 20
        elif 4 < block_id <= 6:
            return min(5, c1+0.1), 25
        elif 6 < block_id <= 8:
            return min(6, c1+0.1), 30
        elif 8 < block_id <= 10:
            return min(7, c1+0.1), 30
        elif 10 < block_id <= 12:
            return min(7.5, c1+0.1), 35
        elif block_id > 12:
            return min(8, c1+0.1), 35

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--AnnealingFlow_config', default = '/GMM_sphere_c=10.yaml', type=str, help='Path to the YAML file')

args_parsed = parser.parse_args()
with open(args_parsed.AnnealingFlow_config, 'r') as file:
    args_yaml = yaml.safe_load(file)

if __name__ == '__main__':
    master_dir = '/storage/home/hcoda1/3/dwu381/scratch/AnnealingFlow_Final_Version'
    Type = args_yaml['data']['type']
    Xdim_flow = args_yaml['data']['Xdim_flow']
    if Type == 'GMM_sphere':
        num_means = args_yaml['data']['num_means']
    elif Type == 'GMM_cube':
        if Xdim_flow <= 5:
            num_means = 2 ** Xdim_flow
        else:
            num_means = 2 ** 5
    block_idxes = args_yaml['training']['block_idxes']
    c = args_yaml['data']['c']
    c1 = c
    master_dir = os.path.join(master_dir, f'samplers_trained/d={Xdim_flow}_{Type}_c={c1}_7')
    for block_id in block_idxes:
        if Type == 'truncated':
            c, punishment = get_c_and_punishment(block_id)
        if Type == 'funnel':
            beta = 1
        elif Type == 'GMM_sphere' and c1 >= 12:
            if block_id <= 8:
                c = 8
            else:
                c = c1
            beta = get_beta(block_id, number = 8)
        elif Type == 'GMM_sphere' and Xdim_flow > 2:
            beta = get_beta(block_id, number = 15)
        elif Type == 'GMM_cube':
            beta = 1
        elif Type == 'exponential':
            if Xdim_flow >= 4:
                beta = get_beta(block_id, number = 20)
            else:
                beta = get_beta(block_id, number = 8)
        else:
            beta = get_beta(block_id, number = 8)
        folder_suffix = args_yaml['eval']['folder_suffix']
        os.makedirs(master_dir, exist_ok=True)
        prefix = 'block'
        common_name = f'{prefix}{block_id}'
        filepath = os.path.join(master_dir, common_name + '.pth')
        directory = os.path.join(master_dir, common_name)
        os.makedirs(directory, exist_ok=True) 
        filename = os.path.join(directory, common_name)
        self = Namespace() 
        print(f'#### Training block {block_id} ####')
        print('########################## Data part ##########################')
        Xdim_flow = args_yaml['data']['Xdim_flow']
        batch_size = args_yaml['training']['batch_size']
        ntr, nte = args_yaml['training']['ntr'], args_yaml['training']['nte']
        xtr = gen_data(ntr, args_yaml['data']['Xdim_flow'])
        xte = gen_data(nte, args_yaml['data']['Xdim_flow'])        
        xraw = xtr.float().to(device)
        xte = xte.float().to(device)
        self.X_test = xte

        if block_id > 1:
            common_name_data = f'{prefix}{block_id-1}'
            filename_data = os.path.join(master_dir, common_name_data + '_Xpushed.pkl')
            Xtrain_pushed = pickle.load(open(filename_data, 'rb'))
            train_loader_raw = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(Xtrain_pushed),
                batch_size=batch_size, shuffle=True)
        else:
            train_loader_raw = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(xraw),
                batch_size=batch_size, shuffle=True)
        train_loader_raw_tr = loop_data_loader(train_loader_raw) # The dataloader

        print('########################## CNF flow setup ##########################')
        vfield_config = Namespace(hid_dims = args_yaml['CNF']['hid_dims'], Xdim_flow = Xdim_flow)
        self.CNF = default_CNF_structure(config = vfield_config)
        total_params = sum(p.numel() for p in self.CNF.parameters())
        print(f'######## Number of parameters in CNF: {total_params/1e3}K ########')
        common_args_CNF = Namespace(
            int_mtd = 'rk4',
            num_e = 1, 
            num_int_pts = 1,  # Number of points for numerical integration
            fix_e_ls = True, 
            use_NeuralODE = True, 
            rtol = 1e-3, #1e-5
            atol = 1e-3 #1e-5
        )
        S = args_yaml['CNF']['S_ls'][block_id-1]
        hk_blocks = args_yaml['CNF']['hk_blocks']
        hk_b = 1
        delta_tk = hk_blocks[block_id-1]
        hk_ls = np.array([hk_b/S] * S)
        self.ls_args_CNF = []
        for i in range(S):
            args_CNF_now = Namespace(**vars(common_args_CNF))  
            hk_sub = hk_ls[i]  # time step of sub-block i
            args_CNF_now.Tk_1 = 0 if i == 0 else np.sum(hk_ls[:i])
            args_CNF_now.Tk = args_CNF_now.Tk_1 + hk_sub
            self.ls_args_CNF.append(args_CNF_now)  
        args_CNF_ = Namespace(**vars(args_CNF_now))
        args_CNF_.Tk_1 = 0
        args_CNF_.Tk = 1         
        for i, a in enumerate(self.ls_args_CNF):
            print(f'##### Sub-Interval {i+1}: [{a.Tk_1}, {a.Tk}], h_k = {a.Tk - a.Tk_1}, m_k = {a.num_int_pts}')
            print(f'Penalty delta_tk at block {block_id} is {delta_tk}')
        print('Done instantiating CNF and CNF args')
        self.block_now = len(self.ls_args_CNF)
        print('########################## Training args ##########################')
        load_checkpoint = args_yaml['training']['load_checkpoint']

        args_training = Namespace(
            tot_iters = args_yaml['training']['tot_iters'], 
            lr = args_yaml['training']['lr'], 
            load_checkpoint = load_checkpoint, 
            iter_start = 0,
            lam_jac = 0, 
        )
        override_default = True 
        optimizer = torch.optim.Adam(self.CNF.parameters(), lr=args_training.lr)
        print('########################## Resume from checkpoint (or not) ##########################')
        self_ls_prev = []
        if block_id > 1:
            print(f'############ Loaded previous CNFs ############')
            load_configs = Namespace(block_id = block_id, master_dir = master_dir, vfield_config = vfield_config,\
                prefix = prefix)
            self_ls_prev = load_prev_CNFs(load_configs)
            assert len(self_ls_prev) == block_id - 1

            if args_yaml['training']['warm_start']:  # True
                self.CNF.load_state_dict(self_ls_prev[-1].CNF.state_dict())
                print(f'############ Warm start from {block_id-1} parameter ############')

        if args_training.load_checkpoint and os.path.exists(filepath):
            checkpt = torch.load(filepath)
            self.CNF.load_state_dict(checkpt['model'])
            args_training = checkpt['args']
            args_training.load_checkpoint = True
            self.loss_at_block = checkpt['loss_at_block']
            optimizer.load_state_dict(checkpt['optimizer'])
            print(f'Starting at batch # {args_training.iter_start+1}')
        else:
            self.loss_at_block = []
            print('Starting from batch # 0')
        self.CNF = torch.nn.DataParallel(self.CNF)  # Distribute the data on each paralleled GPU
        print(self.CNF)
        if override_default:
            args_training.tot_iters = args_yaml['training']['tot_iters']
            print(f'############ Train until {args_training.tot_iters} batches ############')

        # This is to train the current block
        print('########################## Start training ##########################')
        while args_training.iter_start < args_training.tot_iters:
            i = args_training.iter_start
            start = time.time()
            on_off(self, on = False) 
            xsub = next(train_loader_raw_tr)[0]

            # 'add_diffuse' is set as None in args_yaml
            if block_id == 1 and 'add_diffuse' in args_yaml['training']:
                xsub = add_diffuse(xsub)
            optimizer.zero_grad()

            loss_V, loss_div, loss_W2, loss_Jac, _ = JKO_loss_func(xsub, self.CNF, self.ls_args_CNF, beta)
            loss = loss_V + loss_div + loss_W2 + loss_Jac

            if np.isnan(loss.item()):
                raise ValueError('NaN encountered.')
            loss.backward()
            if args_yaml['training']['clip_grad']:
                _ = torch.nn.utils.clip_grad_norm_(self.CNF.parameters(), 1.0)
            optimizer.step()
            args_training.iter_start += 1 
            current_loss = [loss.item(), loss_W2.item(), loss_V.item(), loss_div.item(), loss_Jac.item()]
            self.loss_at_block.append(current_loss)
            if args_training.iter_start % 100 == 0:
                print(f'Iter {args_training.iter_start} with {batch_size} batches done')
            viz_freq = args_yaml['eval']['viz_freq']
            max_iter = args_training.tot_iters - 1

            sdict = {'model': self.CNF.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'args': args_training,
                     'ls_args_CNF': self.ls_args_CNF,
                     'loss_at_block': self.loss_at_block}

            if i % viz_freq == 0 or i == max_iter:
                print(f'######### Evaluate at iter {i+1}')    
                on_off(self, on = True)
                move_configs = Namespace(block_id = block_id, self_ls_prev = self_ls_prev)
                Z_traj = move_over_blocks(self, move_configs, nte = nte) # Towards the target distribution
                index = i//viz_freq
                plot_traj(Z_traj, args_yaml['data']['Xdim_flow'])

                on_off(self, on = False)
                torch.cuda.empty_cache()               
                torch.save(sdict, filepath) # Here we have stored the block's information
            
            if i == max_iter:
                on_off(self, on = True)
                nmax = 10000
                move_configs = Namespace(block_id = block_id, self_ls_prev = self_ls_prev)
                X_traj = move_over_blocks(self, move_configs, nte=nmax)
                Xhat = X_traj[-1].cpu().numpy()
                #mmd_dict = get_MMD(X, Xhat, nmax = nmax)
                #print(mmd_dict)
                on_off(self, on = False)
                Xtrain_pushed = push_samples_forward(train_loader_raw, self)
                print(f'##### Shape of Xtrain_pushed is {Xtrain_pushed.shape} #####')
                filename_data = filepath.split('.pth')[0] + '_Xpushed.pkl'
                pickle.dump(Xtrain_pushed, open(filename_data, 'wb'))
                pickle.dump(Xtrain_pushed, open(filename_data, 'wb'))
                pickle.dump(Xtrain_pushed, open(filename_data, 'wb'))