import os
import argparse
import yaml
import math
import matplotlib.pyplot as plt
import scipy
import pickle
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq as tdeq
import torch.autograd as autograd
import time
import warnings
from scipy.io import mmread
from argparse import Namespace
from scipy.stats import gaussian_kde
import time
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpu = torch.cuda.device_count()
mult_gpu = False if num_gpu < 2 else True

def gen_data(datasize, dim):
    #data = torch.randn(datasize, dim)
    mean = 0
    variance = 1.0
    standard_normal_samples = torch.randn(datasize, dim)
    data = standard_normal_samples * variance + mean
    return data

def gen_squared_data(datasize, dim):
    # Generate uniform random data between 0 and 1
    uniform_samples = torch.rand(datasize, dim)
    # Scale to range (-3, 3)
    data = uniform_samples * 16 - 8
    return data


def get_e_ls(out, num_e):
    e_ls = []
    for i in range(num_e):
        e_ls.append(torch.randn_like(out).to(device))  # returns standard normal
    return e_ls


# Divergence Approximation:
# div(v) = E_p[els*\frac{v(x+\sigma*els)-v(x)}{\sigma}], where els~P=N(0,1)
def divergence_approx(out, x, e_ls=[], t = None, net = None):
    # e_ls: sampled epsilon from N(0,1)
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
    # approx_tr_dzdx_out: primary output, representing the estimated divergence
    # Jac_norm_out: the mean norm of the Jacobian

# Brute-force method to obtain the exact divergence
def divergence_bf(dx, x):
    sum_diag = 0.
    for i in range(x.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(),
                                        x, create_graph=True)[0][:, i]
    return sum_diag.view(x.shape[0], 1)
# This is exact but computationally expensive for high-dimensional X, due to the iteration of dimensions

# This serves as the velocity field
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
    # The inputs and outputs has the same dimension

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
                if self.div_bf: # Exact divergence through brute-force method
                    with torch.set_grad_enabled(True):
                        x.requires_grad_(True)
                        t.requires_grad_(True)
                        out = self.model(x,t)
                        divf = divergence_bf(out, x).to(device)
                        Jac_norm_out = torch.zeros_like(divf).to(device)
                else: # Approximated divergence
                    out = self.model(x,t)
                    divf, Jac_norm_out = divergence_approx(out, x, self.e_ls,
                                                        t = t, net = self.model)
            else:
                divf = torch.zeros(x.shape[0]).to(device)
                Jac_norm_out = torch.zeros_like(divf)
                out = self.model(x,t)
            # -divf is returned
            return out, -divf, Jac_norm_out
            # out denotes x's velocity field value at time t
        return odefunc_wrapper(t, x)

class CNF(nn.Module):
    # Continuous Normalizing Module
    def __init__(self, odefunc):
        super(CNF, self).__init__()
        self.odefunc = odefunc
        # odefunc = ODEFunc(FCnet)
        # odefunc returns: out, -divf, Jac_norm_out
        # out denotes x's velocity field value at time t
    def forward(self, x, args, reverse=False, test=False, mult_gpu=False):
        # In this code, we pass each sub-interval's args of a block continuouslly, to CNF
        self.odefunc.logpx = True
        integration_times = torch.linspace(
            args.Tk_1, args.Tk, args.num_int_pts+1).to(device)
        if test:
            self.odefunc.logpx = False  
        if reverse:
            integration_times = torch.flip(integration_times, [0])
        self.odefunc.num_e = args.num_e
        dlogpx = torch.zeros(x.shape[0]).to(device)
        dJacnorm = torch.zeros(x.shape[0]).to(device)
        self.odefunc.e_ls = None 
        self.odefunc.fix_e_ls = args.fix_e_ls
        self.odefunc.counter = 0

        # !!!!!!!!!!!!!!!
        out = self.odefunc(0,x,velocity=True)

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
        # predz: transformed data through integration of ODE
        # dlogpx: the accumulated divergence
        # dlogpx, dJacnorm: store the log density change and Jacobian norm change, respectively.
        # out: returns the x's velocity field value at t


def default_CNF_structure(config):
    model = FCnet(config).to(device)
    odefunc = ODEFunc(model).to(device)
    CNF_ = CNF(odefunc).to(device)
    return CNF_


def FlowNet_forward(xinput, CNF, ls_args_CNF,
                    block_now,
                    reverse = False, test = True,
                    return_full = False):
    if block_now == 0:
        return xinput, 0
    else:
        ls_args_CNF = ls_args_CNF[:block_now]  # all the sub-intervals' args, containing the start and end time of each sub-interval
        with torch.no_grad():
            predz_ls, dlogpx_ls = [], []
            if reverse:
                ls_args_CNF = list(reversed(ls_args_CNF))
            # ls_args_CNF: time stamps of sub-intervals of a block
            for i, args_CNF in enumerate(ls_args_CNF):
                predz, dlogpx, _, _ = CNF(xinput, args_CNF, 
                                    reverse = reverse, test = test,
                                    mult_gpu = mult_gpu)
                if mult_gpu:
                    if i == 0:
                        predz_ls.append(xinput)
                        dlogpx_ls.append(torch.zeros(xinput.shape[0]).to(device))
                    predz_ls.append(predz)
                    dlogpx_ls.append(dlogpx)
                    xinput = predz
                else:
                    xinput = predz[-1] # The pushed-forward results from previous k-1 blocks
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
        # predz_ls: list of predicted values, at each sub-interval, of one block


# This computes the squared L2 norm of the input tensor.
def l2_norm_sqr(input, return_full = False):
    if len(input.size()) > 2:
        norms = 0.5*input.view(input.shape[0], -1).pow(2).sum(axis=1)
    else:
        norms = 0.5*input.pow(2).sum(axis=1)
    if return_full:
            return norms
    else:
        return norms.mean()


def jacobian_manual_Bayesian_Logistic(input_tensor):
    if gradient_based == True:
        beta = input_tensor[:, :Xdim_flow - 1].to(device)
        alpha = input_tensor[:, -1].to(device) 
        # logistic = torch.sigmoid(X_train_tensor @ beta.T)
        # error = logistic - y_train_tensor.unsqueeze(1)
        # grad_beta = (X_train_tensor.T @ error).T #/ X_train_tensor.shape[0]
        # grad_beta += torch.exp(alpha).unsqueeze(1)*beta
        # grad_alpha = -Xdim_flow/(2*torch.exp(alpha)) + 0.5 * torch.sum(beta**2, dim=1)
        # jacobian_vector = torch.cat([grad_beta, grad_alpha.unsqueeze(1)], dim=1)
        # return jacobian_vector
    else:
        beta = input_tensor[:, :Xdim_flow - 1].to(device)
        alpha = input_tensor[:, -1].to(device) 
        logits = X_train_tensor @ beta.T
        logistic = torch.sigmoid(logits)
        epsl = 1e-8
        nll = -y_train_tensor.unsqueeze(1) * torch.log(logistic+epsl) - (1 - y_train_tensor.unsqueeze(1)) * torch.log(1 - logistic+epsl)
        nll = nll.sum(dim=0)
        prior = 0.5 * torch.exp(alpha) * torch.sum(beta**2, dim=1) + 0.01*torch.exp(alpha) - (Xdim_flow-1)/2 * alpha
        loss =  prior + nll
        avg_loss = loss.mean()
        return avg_loss


def JKO_loss_func(xinput, model, ls_args_CNF):
    num_rk4 = len(ls_args_CNF)
    loss_div_tot, loss_Jac_tot = 0, 0
    xinput_ = xinput.clone()
    for k in range(num_rk4):
        args = ls_args_CNF[k]
        predz, dlogpx, lossJacnorm, v_field = model(xinput, args, reverse = False, test = False,
                                           mult_gpu = mult_gpu) 
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

    loss_V_dot = jacobian_manual_Bayesian_Logistic(xpk)
    return loss_V_dot, loss_div_tot, loss_W2_tot, loss_Jac_tot, raw_movement
    # 1. loss_V_tot: L2 norm of the final transformed data
    # 2. loss_div_tot: Total divergence loss, accumulated over all blocks
    # 3. loss_W2_tot: Wasserstein distance loss, which measures the total movement of the data through the flow
    # loss_Jac_tot: Total Jacobian norm loss, accumulated over all blocks, encouraging the flow to be stable and well-behaved


# Moves data through all given block (forward/backward), and return the intermediate results
# Only used at test time (to see the reversed data)
def move_over_blocks(self, move_configs, reverse = False, nte = 1000):
    with torch.no_grad():
        if reverse:
            Xtest = torch.randn(nte, Xdim_flow).to(device)
        else:
            Xtest = self.X_test.to(device)
        if move_configs.block_id > 1 and reverse is False:
            Zhat_ls_prev, Zout = [Xtest], Xtest
            for self_mod in move_configs.self_ls_prev:
                Zout, _ = FlowNet_forward(Zout, self_mod.CNF, self_mod.ls_args_CNF, #ls_args_CNF consists of all sub-int
                                          self_mod.block_now, reverse = False,
                                          return_full = False)
                Zhat_ls_prev.append(Zout)
            Xtest = Zout
        # Begin current block:
        Zhat_ls, _ = FlowNet_forward(Xtest, self.CNF, self.ls_args_CNF,
                                     self.block_now,
                                     reverse = reverse,
                                     return_full = True)
        if move_configs.block_id > 1 and reverse is True:
            Xhat_ls_prev, Xout = [], Zhat_ls[-1]
            for self_mod in reversed(move_configs.self_ls_prev):
                Xout, _ = FlowNet_forward(Xout, self_mod.CNF, self_mod.ls_args_CNF,
                                          self_mod.block_now, reverse = True,
                                          return_full = False)
                Xhat_ls_prev.append(Xout)
        if mult_gpu:
            ids = range(len(Zhat_ls))
        else:
            ids = torch.linspace(0, Zhat_ls.shape[0]-1, self.block_now+1).long()
        Zhat_ls = [Zhat_ls[i] for i in ids]
        if move_configs.block_id > 1:
            Zhat_ls = Zhat_ls_prev + Zhat_ls[1:] if reverse is False else Zhat_ls + Xhat_ls_prev
    return Zhat_ls 

def push_samples_forward(data_loader, self):
    X = []
    for xsample in data_loader:
        xsample = xsample[0]
        xpushed, _ = FlowNet_forward(xsample.to(device), self.CNF, 
                                    self.ls_args_CNF, self.block_now,
                                    reverse = False, test = True, 
                                    return_full = False)
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

def pdist(sample_1, sample_2, norm=2):
    return torch.cdist(sample_1, sample_2, p=norm)

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


def Bayesian_Logistic_test(Ztraj, X_test, y_test):
    samples = Ztraj[-1]
    if gradient_based == True:
        samples = samples[:, :Xdim_flow-1]
    else:
        samples = samples[:, :Xdim_flow-1]

    # Calculate predictions and accuracies for each sample
    accuracies = []
    posteriors = []
    for weight in samples:
        logits = X_test_tensor @ weight
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float()
        accuracy = (predictions == y_test_tensor).float().mean().item()
        accuracies.append(accuracy)
        prob = torch.where(y_test_tensor == 1, probs, 1 - probs)
        posteriors.append(prob)
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    avg_posteriors = torch.stack(posteriors).mean(dim=0)
    log_avg_posteriors = torch.log(avg_posteriors)
    avg_log_posterior = log_avg_posteriors.mean().item()

    print(f"Average test accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average log posterior: {avg_log_posterior:.4f}")

    return avg_accuracy, std_accuracy, avg_log_posterior

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--JKO_config', default = 'Annealing_Flow.yaml', type=str, help='Path to the YAML file')
args_parsed = parser.parse_args()
with open(args_parsed.JKO_config, 'r') as file:
    args_yaml = yaml.safe_load(file)

if __name__ == '__main__':
    block_idxes = args_yaml['training']['block_idxes']
    gradient_based = False
    dataset = args_yaml['data']['dataset']
    file_path = f'dataset/{dataset}'

    if dataset == 'codon_usage.csv':
        df = pd.read_csv(file_path, header=0, low_memory=False)  # Suppress mixed type warnings
        df = df[(df['DNAtype'] == 0) | (df['DNAtype'] == 1)]  # Keep rows where DNAtype == 0 or 1
        numeric_df = df.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
        numeric_df = numeric_df.dropna()  # Drop rows with any NaN values
        filtered_df = df.loc[numeric_df.index]  # Align with clean rows
        y = torch.tensor(filtered_df.iloc[:, 1].values, dtype=torch.float32)  # Second column as y
        X = torch.tensor(numeric_df.values, dtype=torch.float32)  # Columns 6 to the end as X
        print(X.shape)
        print(y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    else:
        X, y = load_svmlight_file(file_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32).to(device)


    # Add a column of ones to both training and test tensors
    X_train_tensor = torch.cat([X_train_tensor, torch.ones(X_train_tensor.shape[0], 1, device=device)], dim=1)
    X_test_tensor = torch.cat([X_test_tensor, torch.ones(X_test_tensor.shape[0], 1, device=device)], dim=1)


    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.where(y_train_tensor == -1, torch.tensor(0., device=device), y_train_tensor)
    y_test_tensor = torch.where(y_test_tensor == -1, torch.tensor(0., device=device), y_test_tensor)
    # Check if labels are not 0/1
    unique_labels = torch.unique(y_train_tensor)
    if not torch.all(torch.isin(unique_labels, torch.tensor([0., 1.], device=device))):
        print("Labels are not 0/1. Converting to 0/1...")
        min_label = torch.min(y_train_tensor)
        max_label = torch.max(y_train_tensor)
        y_train_tensor = torch.where(y_train_tensor == min_label, torch.tensor(0., device=device), torch.tensor(1., device=device))
        y_test_tensor = torch.where(y_test_tensor == min_label, torch.tensor(0., device=device), torch.tensor(1., device=device))
        print(f"Converted labels: {min_label.item()} -> 0, {max_label.item()} -> 1")
    if gradient_based == True:
        Xdim_flow = X_train_tensor.shape[1] + 1
    else:
        Xdim_flow = X_train_tensor.shape[1] + 1

    master_dir = f'results/{dataset}'
    os.makedirs(master_dir, exist_ok=True)
    acc_list = []
    for block_id in block_idxes:
        folder_suffix = args_yaml['eval']['folder_suffix']
        os.makedirs(master_dir, exist_ok=True)
        prefix = 'block'
        common_name = f'{prefix}{block_id}'
        filepath = os.path.join(master_dir, common_name + '.pth') # Store the block in .pth file
        directory = os.path.join(master_dir, common_name)
        os.makedirs(directory, exist_ok=True) 
        filename = os.path.join(directory, common_name)
        self = Namespace() 
        print(f'#### Training block {block_id} ####')
        print('########################## Data part ##########################')
        #Xdim_flow = args_yaml['data']['Xdim_flow']
        batch_size = args_yaml['training']['batch_size'] # Batch Size
        ntr, nte = args_yaml['training']['ntr'], args_yaml['training']['nte']
        xtr = gen_data(ntr, Xdim_flow)
        xte = gen_data(nte, Xdim_flow)
        xraw = xtr.float().to(device)
        xte = xte.float().to(device)
        self.X_test = xte

        #Initialize the data loader. If not the first block, load the data from the previous block's output.
        if block_id > 1:
            common_name_data = f'{prefix}{block_id-1}'
            # finename_data is to store the pushed X data
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
            num_int_pts = 1,  # Number of integrations
            fix_e_ls = True, 
            use_NeuralODE = True, 
            rtol = 1e-3, #1e-5
            atol = 1e-3 #1e-5
        )
        S = args_yaml['CNF']['S_ls'][block_id-1] # 3
        hk_blocks = args_yaml['CNF']['hk_blocks'] # penalty
        hk_b = 1  # The time-duration for each sub-block is set as 1
        delta_tk = hk_blocks[block_id-1]  # Penalty at outer block k
        hk_ls = np.array([hk_b/S] * S) # time step-sizes for each sub-block
        self.ls_args_CNF = []  # self is a Namespace
        for i in range(S): # Iterate over each sub-interval
            args_CNF_now = Namespace(**vars(common_args_CNF))  
            hk_sub = hk_ls[i]  # time step of sub-block i
            args_CNF_now.Tk_1 = 0 if i == 0 else np.sum(hk_ls[:i]) # set to 0 for the first sub-block
            args_CNF_now.Tk = args_CNF_now.Tk_1 + hk_sub  # the end time for the current sub-block
            
            self.ls_args_CNF.append(args_CNF_now)  
        args_CNF_ = Namespace(**vars(args_CNF_now))
        args_CNF_.Tk_1 = 0
        args_CNF_.Tk = 1         
        for i, a in enumerate(self.ls_args_CNF):
            print(f'##### Sub-Interval {i+1}: [{a.Tk_1}, {a.Tk}], h_k = {a.Tk - a.Tk_1}, m_k = {a.num_int_pts}')
            print(f'Penalty delta_tk at block {block_id} is {delta_tk}')
        print('Done instantiating CNF and CNF args')

        self.block_now = len(self.ls_args_CNF)  # Number of sub-intervals
        print('########################## Training args ##########################')
        load_checkpoint = args_yaml['training']['load_checkpoint']  # Boolean: True
        
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
            on_off(self, on = False) 
            xsub = next(train_loader_raw_tr)[0]
            end = time.time()

            optimizer.zero_grad()
            loss_V, loss_div, loss_W2, loss_Jac, _ = JKO_loss_func(xsub, self.CNF, self.ls_args_CNF)
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
                Z_traj = move_over_blocks(self, move_configs, reverse = False, nte = nte) # Towards the target distribution
                index = i//viz_freq
                acc, std, _ = Bayesian_Logistic_test(Z_traj, X_test_tensor, y_test_tensor)
                print(f'Test accuracy: {acc:.4f} ± {std:.4f}')
                acc_list.append([block_id, acc, std])
                
                on_off(self, on = False)
                torch.cuda.empty_cache()               
                torch.save(sdict, filepath)
                helper(on = True)
                helper(on = False)
            
            if i == max_iter:
                on_off(self, on = True)
                nmax = 10000
                move_configs = Namespace(block_id = block_id, self_ls_prev = self_ls_prev)
                X_traj = move_over_blocks(self, move_configs, reverse = True, nte=nmax)
                Xhat = X_traj[-1].cpu().numpy()
                on_off(self, on = False)
                Xtrain_pushed = push_samples_forward(train_loader_raw, self)
                print(f'##### Shape of Xtrain_pushed is {Xtrain_pushed.shape} #####')
                filename_data = filepath.split('.pth')[0] + '_Xpushed.pkl'
                pickle.dump(Xtrain_pushed, open(filename_data, 'wb'))