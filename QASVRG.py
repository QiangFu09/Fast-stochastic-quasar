# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:49:49 2022


"""
import pickle
import torch
import random
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F
import argparse
import torch.nn as nn
import itertools
from itertools import zip_longest
SQRT_FLT_EPS = torch.tensor(np.sqrt(np.finfo(np.float32).eps))
SQRT_DBL_EPS = torch.tensor(np.sqrt(np.finfo(np.float64).eps), dtype=torch.double)
'''
#dataset1
df = np.loadtxt('banknote.txt',delimiter=',')
df = torch.from_numpy(df)
df_row = df.size(dim=0)
df_col = df.size(dim=1)
feature = df[0:df_row, 0:df_col-1]
label = 2 * (df[0:df_row, -1] - 0.5)
feature = feature.float()
label = label.float()
feature = F.normalize(feature, p=2, dim=1)
'''

def poly_mult(p1, p2):
    # input: coefficient lists, in order of increasing power
    # output: coefficient list of p1*p2
    p3 = torch.zeros(len(p1) + len(p2) - 1)
    for power, coef in enumerate(p1):
        p2_shifted = torch.cat((torch.zeros(power), p2, torch.zeros(len(p3) - len(p2) - power)))
        p3 += coef*p2_shifted
    return p3

def update_h(A_lastrow, h, x_cur):
    out = torch.empty_like(h)
    out[:, :-1] = h[:, 1:]
    out[:, -1] = torch.addmm(x_cur, h, A_lastrow).view(-1)
    return out

def simulate_helper(A_lastrow, C, D, x, h=None):
    b, T = x.size()
    n = len(C)
    if h is None:
        h = torch.zeros(b, n).to(dtype=x.dtype, device=x.device)
    assert (len(A_lastrow) == h.size(1) == n)
    A_lastrow, C = A_lastrow.view(-1, 1), C.view(-1, 1)
    y = torch.empty(b, T).to(dtype=x.dtype, device=x.device)
    for i in range(T):
        x_cur = x[:, i:i+1]
        h_new = update_h(A_lastrow, h, x_cur)
        y[:, i] = torch.addmm(D*x_cur, h, C).view(-1)
        h = h_new
    return y

def simulate(params_cat, x, h=None, no_grad=False):
    n = len(params_cat) // 2
    assert (len(params_cat) == 2*n+1)
    A_lastrow = params_cat[:n]
    C = params_cat[n:2*n]
    D = params_cat[-1]
    if no_grad:
        with torch.no_grad():
            return simulate_helper(A_lastrow, C, D, x, h)
    return simulate_helper(A_lastrow, C, D, x, h)

args_list = None
parser = argparse.ArgumentParser('Linear Dynamical Systems Experiments')
parser.add_argument('--n', type=int, default=20, help='Hidden state size')
parser.add_argument('--T', type=int, default=500, help='Trajectory length')
parser.add_argument('--N', type=int, default=400000, help='Maximum number of optimization steps')
parser.add_argument('--tol', type=float, default=1e-4, help='Gradient norm termination criterion (0 for none)')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--print-every', type=int, default=10, help='Print every X steps')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--b', type=int, default=5000, help='Batch size; also number of unique samples')
parser.add_argument('--noise-scale', type=float, default=0.1, help='Noise scale for perturbing initial guess')
parser.add_argument('--radius', type=float, default=0.95, help='Radius of circle in which to select char poly eigs')
parser.add_argument('--optimizer', type=str, choices=['gd', 'agd', 'ours'], default='gd', help='Which optimizer to use')
parser.add_argument('--gamma', type=float, default=1., help='Gamma value (for our method)')
parser.add_argument('--finite-diff', '--fd', action='store_true', help='Use finite difference gradients in line search')
parser.add_argument('--guess', type=int, choices=[0, 1, 2], default=0, help='Try convex alpha as a guess (0 for no, 1 for yes, 2 to try it first)')
parser.add_argument('--double', action='store_true', help='Use double precision')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--progress', action='store_true', help='Display progress bar')
args = parser.parse_args(args_list)
n, T, N, lr, print_every, seed, b, noise_scale, radius, gamma = args.n, args.T, args.N, args.lr, args.print_every, args.seed, args.b, args.noise_scale, args.radius, args.gamma
median_r = 17
T1 = T//4
torch.manual_seed(12)

r_too_big = True
while r_too_big:
    phases = torch.FloatTensor(n//2).uniform_(0, 2*np.pi)
    magnitudes = torch.FloatTensor(n//2).uniform_(0, radius)
    real_parts = magnitudes * torch.sin(phases)
    poly = torch.tensor([1.])
    for j in range(n//2):
        # conjugate pair (x-(a+bi))(x-(a-bi)) = x^2 - 2*a*x + (a^2+b^2)
        conjpair_poly = torch.tensor([magnitudes[j]**2, -2*real_parts[j], 1.])
        poly = poly_mult(poly, conjpair_poly)

    A_lastrow = -poly[:-1]
    C = torch.randn(n)
    A = torch.zeros(n, n)
    A[:-1, 1:] = torch.eye(n-1)
    A[-1, :] = A_lastrow
    At = torch.eye(n)
    r = torch.empty(T+1)
    for j in range(T+1):
        r[j] = torch.dot(C, At[:, -1])
        At = torch.matmul(A, At)
    if torch.norm(r).item() < median_r * 10:
        r_too_big = False

D = torch.tensor(1.).normal_()
feature = torch.randn(b, T) #feature
h0s = torch.randn(b, n)
true_cat = torch.cat([A_lastrow, C, torch.tensor([D])])
label = simulate(true_cat, feature, h0s, no_grad=True) #label

class SVRG_options:
    def __init__(self):
        self.max_iters = 2000
        self.max_stages = 10000
        self.tol = 1e-2 #tol=1e-3 when \mu > 0, tol=1e-2 when \mu = 0.
        self.tol_type = 'func'
        self.lr = 1e-6 #3e-5 for qasvrg(0) 1e-6(24) 3e-6(12)
        self.step_type = 'adaptive'
        self.step_inc = 1.1
        self.step_dec = 0.6
        self.feature = feature
        self.label = label
        self.max_ss_iters = 100
        self.max_as_iters = 100
        self.reduced_tau = True
        self.restart = 'none'
        self.num_tol = 1e-8
        self.mode = 'standard'
        self.ls_mode = 'exact'
        self.ls_guess = True
        self.guess_first = True
        self.verbose = True
        self.alpha = 0.8   #We specify the value of gamma here {0.5, 0.8}
        self.q = 1/4
        self.b = 0
        self.mu = 1/500
    def get_minibatch(self, k): #minibatch for QASVRG(non-strong)
        nu = self.alpha*len(self.feature)*(2*k+3)
        de = self.alpha*(len(self.feature)-1)/8+self.alpha*(2*k+3)
        b = math.ceil(nu/de)
        full_index = list(range(len(feature)))
        selected_index = torch.tensor(random.sample(full_index, b))
        feature_mini = self.feature[selected_index]
        label_mini = self.label[selected_index]
        return feature_mini, label_mini
    def get_minibatch_str(self): #minibatch for QASVRG(strong, Option 1)
        kappa = self.lr / self.mu
        nu = 8*len(self.feature)*((8*kappa)**0.5+self.alpha)
        de = (len(self.feature)-1)*self.alpha + 8*((8*kappa)**0.5+self.alpha)
        b = math.ceil(nu/de)
        full_index = list(range(len(feature)))
        selected_index = torch.tensor(random.sample(full_index, b))
        feature_mini = self.feature[selected_index]
        label_mini = self.label[selected_index]
        return feature_mini, label_mini

def ssq(v):
    return torch.sum(v*v)

def incr(alpha, h):
    temp = alpha + h
    h = temp - alpha
    return temp, h

def binary_search(fg, fe, la, para, x, v, L, b, c, fx, eps, max_iters, reduced_tau,
                  grad_mode, guess=None, guess_first=False):
    fd = (grad_mode != 'exact')
    num_eps = SQRT_FLT_EPS if x.dtype == torch.float else SQRT_DBL_EPS
    def g(alpha, func_only=False, fval=None):
        assert (0 <= alpha <= 1)
        w = alpha*x + (1-alpha)*v
        if func_only:
            return fg(w, fe, la, para, func_only=True)
        if not fd:
            if fval is not None:
                G_f = fg(w, fe, la, para, grad_only=True)
            else:
                fval, G_f = fg(w, fe, la, para)
            dg = torch.dot(G_f, x-v)
        else:
            fval = fval if fval is not None else fg(w, fe, la, para, func_only=True)
            alpha2, h = incr(alpha, num_eps*alpha)
            w2 = alpha2*x + (1-alpha2)*v
            f2val = fg(w2, fe, la, para, func_only=True)
            dg = (f2val - fval) / h
        return fval, dg, None if fd else G_f
    xv_sqdist = ssq(x-v)
    if xv_sqdist < num_eps**2 or torch.norm((x-v)/x, float('inf')) < num_eps:
        return 1, fx, None  # avoid line search if x, v very close
    p = b*xv_sqdist
    if guess_first and guess is not None and guess != 1:
        g_1 = fx
        g_guess, dg_guess, G_guess = g(guess)
        if c*g_guess + guess*(dg_guess - guess*p) <= c*g_1 + eps:
            return guess, g_guess, G_guess
    g_1, dg_1, G_1 = g(1, fval=fx)
    if dg_1 <= eps + p:
        return 1, g_1, G_1
    g_0 = g(0, func_only=True)
    if c == 0 or g_0 <= g_1 + eps/c:
        return 0, g_0, None
    
    if not guess_first and guess is not None:
        g_guess, dg_guess, G_guess = g(guess)
        if c*g_guess + guess*(dg_guess - guess*p) <= c*g_1 + eps:
            return guess, g_guess, G_guess
    
    if reduced_tau:
        tau = 1 - (eps+p) / (L*xv_sqdist)
        g_tau, dg_tau, G_tau = g(tau)
    else:
        tau = 1
        g_tau, dg_tau, G_tau = g_1, dg_1, G_1
    tau = torch.tensor(tau, dtype=x.dtype)
    alpha, g_alpha, dg_alpha, G_alpha = tau, g_tau, dg_tau, G_tau
    lo, hi = torch.tensor(0, dtype=x.dtype), tau
    n_iters = 0
    while c*g_alpha + alpha*(dg_alpha - alpha*p) > c*g_1 + eps and n_iters < max_iters:
        alpha = (lo + hi) / 2
        g_alpha, dg_alpha, G_alpha = g(alpha)
        if g_alpha <= g_tau:
            hi = alpha
        else:
            lo = alpha
        n_iters += 1
    print(n_iters)
    return alpha.item(), g_alpha, G_alpha    

def phi(x,alpha,func_only=False,grad_only=False):
    if x.item() >= 0 and x.item() <= 1:
        fval = x**2/2
        dfval = x
    elif x.item() > 1:
        fval = (x**alpha-1)/alpha+1/2
        dfval = x**(alpha-1)
    else:
        fval = torch.tensor([0])
        dfval = torch.tensor([0])
    if func_only:
        return fval.item()
    if grad_only:
        return dfval
    return fval.item(), dfval

'''
Our objective function without regularizer; the function will output function value or
gradient or both, which is adjusted by func_only and grad_only
'''
def Hingeloss(x, feature, label, alpha, func_only=False, grad_only=False):
    #feature should be a matrix and label should be a col vector.
    Efval = 0
    Edfval = torch.zeros(feature.size(dim=1), dtype=feature.dtype)
    for i in range(len(feature)):
        fval, dfval = phi(1-label[i]*torch.dot(feature[i], x), alpha)##
        Efval += fval
        Edfval += dfval*label[i]*feature[i]*(-1)
    Efval = Efval / len(feature)
    Edfval = Edfval / len(feature)
    if func_only:
        return Efval
    if grad_only:
        return Edfval
    return Efval, Edfval
'''
Our objective function with regularizer (\mu = 1/500)
'''
def ReHingeloss(x, feature, label, alpha, func_only=False, grad_only=False):
    #feature should be a matrix and label should be a col vector.
    Efval = 0
    Edfval = torch.zeros(feature.size(dim=1), dtype=feature.dtype)
    for i in range(len(feature)):
        fval, dfval = phi(label[i]*torch.dot(feature[i], x), alpha)
        Efval += fval
        Edfval += dfval*label[i]*feature[i]
    Efval = Efval / len(feature)
    Edfval = Edfval / len(feature)
    Efval = Efval + (10 ** (-3)) * ssq(x).item()
    Edfval = Edfval + (1 / 500) * x
    if func_only:
        return Efval
    if grad_only:
        return Edfval
    return Efval, Edfval 

def Log(x, alpha, func_only=False, grad_only=False): # \log(1+e^x)
    fval = math.log(1 + math.exp(x))
    dfval = math.exp(x) / (1 + math.exp(x))
    if func_only:
        return fval.item()
    if grad_only:
        return dfval
    return fval.item(), dfval
    

def Logistic(x, feature, label, alpha, func_only=False, grad_only=False):
    #feature should be a matrix and label should be a col vector.
    Efval = 0
    Edfval = torch.zeros(feature.size(dim=1), dtype=feature.dtype)
    for i in range(len(feature)):
        fval, dfval = phi(-label[i]*torch.dot(feature[i], x), alpha)
        Efval += fval
        Edfval += dfval*label[i]*feature[i]*(-1)
    Efval = Efval / len(feature)
    Edfval = Edfval / len(feature)
    if func_only:
        return Efval
    if grad_only:
        return Edfval
    return Efval, Edfval
'''
Objective of LDS
'''
def func_wrapper(param_vals, feature, label, alpha, func_only=False, grad_only=False):
    if not func_only: param_vals = nn.Parameter(param_vals)
    err = loss(simulate(param_vals, feature, no_grad=func_only)[:, T1:], label[:, T1:])
    if func_only: return err.item()
    if param_vals.grad is not None:
        param_vals.grad.detach_()
        param_vals.grad.zero_()
    err.backward()
    if grad_only: return param_vals.grad.data
    return err.item(), param_vals.grad.data


'''
Some parameter of QASVRG (general quasar-convex)
'''
def alphafun(gamma, L, k): #stepsize of mirror descent in QASVRG(non strong)
    return (gamma*(2*k+3))/(8*L)

def taufun(k):             #guess value of momentum(non strong)
    return 1 - (2*k+3) / (k+2)**2

def rhofun(L, k):          #stepsize of gradient descent
    return 1/L



'''
QASVRG for general quasar-convex functions or strongly quasar-convex functions (Option 2)
'''
def ASVRG(fg, x0, alphafun, taufun, rhofun, options=SVRG_options()):
    options.reduced_tau = False
    K = options.max_stages
    K2 = options.max_iters
    step_size = options.lr
    evals = [0, 0]
    fg_old = fg
    alpha = options.alpha
    gamma = alpha
    #mu = 1/500
    def fg(x, feature, label, alpha, func_only=False, grad_only=False):
        if not grad_only:
            evals[0] += len(feature)
        if not func_only:
            evals[1] += len(feature)
        #evals[0] += not grad_only*len(alpha)
        #evals[1] += not func_only*len(alpha)
        if grad_only:
            return fg_old(x, feature, label, alpha)[1]
        return fg_old(x, feature, label, alpha, func_only=func_only)
    
    L = 1 / step_size
    feature = options.feature
    label = options.label
    omega, z, y = x0, x0, x0
    q = options.q
    b = options.b
    f_y0 = fg_old(y, feature, label, alpha, func_only=True)
    fvals = [f_y0]
    print(f_y0)
    subrout = 0
    num_nostep = 0
    for s in range(K):
        if options.verbose: print(f'Stage {s}')
        full_grad_fixed = fg(omega, feature, label, alpha, grad_only=True)
        fval_diff = fg_old(omega, feature, label, alpha, func_only=True)
        t_nu = (gamma**2/16+1)*0.5*ssq(z)
        t_de = (gamma**2/(16*L))*q*fval_diff
        t = math.ceil((t_nu/t_de)**0.5)#calculate the maximum number of iteration per stage
        b_k = 0
        tol = options.tol*gamma*fg(omega, feature, label, alpha, func_only=True)/2
        for k in range(t):
            subrout += 1
            break2 = False
            if options.verbose: print(f'Step {k}')
            c = gamma*(k+1)**2/(4*k+6) #parameter of Bisearch
            sel_feature, sel_label = options.get_minibatch(k)#get minibatch
            #tol = options.tol*gamma*fg_old(omega, sel_feature, sel_label, alpha, func_only=True)
            if options.verbose: print(len(sel_feature))
            b_k += len(sel_feature)
            fz = fg(z, sel_feature, sel_label, alpha, func_only=True)
            tau, g_tau, G_tau = binary_search(fg, sel_feature, sel_label, alpha, y, z, L, b, c, fz, tol, options.max_as_iters, options.reduced_tau,
                  options.ls_mode, guess=taufun(k) if options.ls_guess else None, guess_first=options.guess_first)
            print(tau)
            #tau = 0.5
            x = (1-tau)*z + tau*y ##momentum
            
            dx = G_tau if G_tau is not None else fg(x, sel_feature, sel_label, alpha, grad_only=True)
            
            grad_estimate = dx - fg_old(omega, sel_feature, sel_label, alpha, grad_only=True) + full_grad_fixed
            
            z_new = z - alphafun(gamma, L, k)*grad_estimate ## mirror descent
            
            y_new = x - rhofun(L,k) * grad_estimate ## gradient descent
            
            fy = fg_old(y_new, feature, label, alpha, func_only=True)
            if options.verbose: print(f'Loss: {fy}')
            if options.tol and options.tol_type == 'func' and fy < options.tol:
                fvals.append(fy)
                break2 = True
                break
            
            if torch.dot(grad_estimate, y_new-y) > 0:##restart criterion
                break ##restart
            
            fvals.append(fy)
            y, z = y_new, z_new
            if len(fvals) >= 2000:
                break2 = True
                break
        omega, z, y = y_new, y_new, y_new
        if break2:
            break
    func_eval, grad_eval = evals
    total_iter = subrout
    return s+1, total_iter, func_eval, grad_eval, fvals, y


'''
QASVRG for strongly quasar-convex functions (Option 1 with fixed batchsize). The choice
of parameter has been put inside this function.
'''
def strASVRG(fg, x0, options=SVRG_options()): #For strongly quasar-convex functions
    K = options.max_stages
    step_size = options.lr
    evals = [0, 0]
    fg_old = fg
    alpha = options.alpha
    gamma = alpha
    mu = options.mu
    def fg(x, feature, label, alpha, func_only=False, grad_only=False):
        if not grad_only:
            evals[0] += len(feature)
        if not func_only:
            evals[1] += len(feature)
        #evals[0] += not grad_only*len(alpha)
        #evals[1] += not func_only*len(alpha)
        if grad_only:
            return fg_old(x, feature, label, alpha)[1]
        return fg_old(x, feature, label, alpha, func_only=func_only)
    
    L = 1 / step_size
    feature = options.feature
    label = options.label
    omega, z, y = x0, x0, x0
    q = options.q
    kappa = L / mu
    b = gamma*mu/4 #parameter of Bisearch
    c = (2*kappa)**0.5 #parameter of Bisearch
    f_y0 = fg_old(y, feature, label, alpha, func_only=True)
    fvals = [f_y0]
    subrout = 0
    tol = 0
    base = 1 + gamma/(8*kappa)**0.5
    exp = 2 / (gamma*q)
    t = math.log(exp, base)
    alpha_k = gamma*mu/2 #parameter of mirror descent
    beta_k = (2*L*mu)**0.5#parameter of mirror descent
    ## we have two choice for guess values in Bisearch
    #str_guess = None #guess can be None
    #str_guess = gamma / (2*gamma+(8*kappa)**0.5) #for complex dataset
    str_guess = 1e-7 
    for s in range(K):
        if options.verbose: print(f'Stage {s}')
        full_grad_fixed = fg(omega, feature, label, alpha, grad_only=True)
        t = math.ceil(t)
        b_k = 0
        for k in range(t):
            subrout += 1
            break2 = False
            if options.verbose: print(f'Step {k}')
            sel_feature, sel_label = options.get_minibatch_str()
            if options.verbose: print(len(sel_feature))
            b_k += len(sel_feature)
            fz = fg(z, sel_feature, sel_label, alpha, func_only=True)
            tau, g_tau, G_tau = binary_search(fg, sel_feature, sel_label, alpha, y, z, L, b, c, fz, tol, options.max_as_iters, options.reduced_tau,
                  options.ls_mode, guess=str_guess if options.ls_guess else None, guess_first=options.guess_first)
            print(tau)
            #tau = 0.5
            x = tau*y + (1-tau)*z##momentum
            #x = (1-tau)*z + tau*y
            dx = G_tau if G_tau is not None else fg(x, sel_feature, sel_label, alpha, grad_only=True)
            
            grad_estimate = dx - fg_old(omega, sel_feature, sel_label, alpha, grad_only=True) + full_grad_fixed
            
            z_new = beta_k*z/(alpha_k+beta_k) + alpha_k*x/(alpha_k+beta_k) - grad_estimate/(alpha_k+beta_k) ## mirror descent
            
            y_new = x - grad_estimate/L ## gradient descent
            
            fy = fg_old(y_new, feature, label, alpha, func_only=True)
            if options.verbose: print(f'Loss: {fy}')
            if options.tol and options.tol_type == 'func' and fy < options.tol:
                fvals.append(fy)
                break2 = True
                break
            fvals.append(fy)
            if torch.dot(grad_estimate, y_new-y) > 0:
                break ##restart
            
            y, z = y_new, z_new
            if subrout >= 500:
                break2 = True
                break
        omega, z, y = y_new, y_new, y_new
        if break2:
            break
    func_eval, grad_eval = evals
    total_iter = subrout
    return s+1, total_iter, func_eval, grad_eval, fvals, y

eig_too_big = True
while eig_too_big:
    noise_Ahat = torch.randn_like(A_lastrow)
    noise_Ahat *= torch.norm(A_lastrow)*torch.rand(1)*noise_scale / torch.norm(noise_Ahat)
    Ahat = A_lastrow + noise_Ahat
    Aa = torch.zeros(n, n)
    Aa[:-1, 1:] = torch.eye(n-1)
    Aa[-1, :] = Ahat.data
    if np.amax(np.absolute(np.linalg.eig(Aa.numpy())[0])) < 0.98:
        eig_too_big = False

Ahat = Ahat
noise_Chat = torch.randn_like(C)
noise_Chat *= torch.norm(C)*torch.rand(1)*noise_scale / torch.norm(noise_Chat)
Chat = C + noise_Chat
Dhat = D + torch.abs(D)*(torch.rand_like(D)*2-1)*noise_scale
params_cat = nn.Parameter(torch.cat([Ahat, Chat, torch.tensor([Dhat])]))
loss = nn.MSELoss()

#x0 = torch.tensor([-65.8015, -23.1198, -92.1828, -13.7849])#Initial point of Dataset 1

options = SVRG_options()
stage1, iters1, func1, grad1, fvals1, y1 = ASVRG(func_wrapper, params_cat.data, alphafun, taufun, rhofun)
stage2, iters2, func2, grad2, fvals2, y2 = ASVRG(func_wrapper, params_cat.data, alphafun, taufun, rhofun)
stage3, iters3, func3, grad3, fvals3, y3 = ASVRG(func_wrapper, params_cat.data, alphafun, taufun, rhofun)
print((func1+grad1+func2+grad2+func3+grad3)/3)
'''
Generate error bar
'''
av = [np.nanmean(x) for x in zip_longest(fvals1,fvals2,fvals3, fillvalue=np.nan)]
avplus = [np.nanmax(x) for x in zip_longest(fvals1,fvals2,fvals3, fillvalue=np.nan)]
avminus = [np.nanmin(x) for x in zip_longest(fvals1,fvals2,fvals3, fillvalue=np.nan)]
data = [avplus, av, avminus]



##save the data of function value per iteration.
'''
with open("SVRGLDS3gamma=0.5+", "wb") as fp: 
    pickle.dump(data, fp)
'''
#print(func+grad)
