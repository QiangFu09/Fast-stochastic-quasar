'''
The code of QAGD is nearly the same as Hinder's code; The only difference is that
we do not use adaptive stepsize!
'''
import numpy as np
import torch
import random
import math
import pickle
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import argparse

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

#Output of the true LDS.
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
torch.manual_seed(36) #0, 24, 48

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

class AGD_options:
    def __init__(self):
        self.max_iters = 1000
        self.tol = 1e-2 #tol=1e-3 when \mu > 0, tol=1e-2 when \mu = 0.
        self.tol_type = 'func'
        self.lr = 1e-2
        self.step_type = 'adaptive' #constant / adaptive
        self.step_inc = 1.1
        self.step_dec = 0.6
        self.max_ss_iters = 100
        self.max_as_iters = 100
        self.feature = feature
        self.label = label
        self.reduced_tau = True
        self.restart = 'none'
        self.num_tol = 1e-8
        self.mode = 'standard'
        self.ls_mode = 'exact' #exact
        self.ls_guess = False
        self.guess_first = False
        self.verbose = True
        self.alpha = 0.8 #We specify the value of gamma here {0.5, 0.8}
    def stochastic_choose(self): #Choose stochastic gradient
        index = random.sample(list(range(len(self.feature))),1)
        return feature[index], label[index]
def ssq(v):
    return torch.sum(v*v)

def incr(alpha, h):
    temp = alpha + h
    h = temp - alpha
    return temp, h

def binary_search(fg, fe, la, para, x, v, L, b, c, fx, eps, max_iters, reduced_tau,
                  grad_mode, guess=None, guess_first=False):
    fd = grad_mode != 'exact'
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
    return alpha.item(), g_alpha, G_alpha

def agd_framework(fg, x0, beta, etafun, cfun, taufun, ls_tol, options):
    if options.step_type == 'search':
        raise NotImplementedError('Full step size search not implemented!')
    K = options.max_iters
    alpha = options.alpha
    step_size = options.lr
    evals = [0, 0]
    fg_old = fg
    def fg(x, feature, label, alpha, func_only=False, grad_only=False):
        if not grad_only:
            evals[0] += len(feature)
        if not func_only:
            evals[1] += len(feature)
        #evals[0] += not grad_only
        #evals[1] += not func_only
        if grad_only:
            return fg_old(x, feature, label, alpha)[1]
        return fg_old(x, feature, label, alpha, func_only=func_only)
    feature = options.feature
    label = options.label
    L = 1 / step_size
    eta = etafun(L, 0)
    x, y, v = x0, x0, x0
    f_y, df_y = 0, 0
    f_x_new = fg(x, feature, label, alpha, func_only=True)
    print(f_x_new)
    fvals = [f_x_new]

    take_step = True
    num_nostep = 0
    k = 0
    while k < K:
        if options.verbose: print(f'Step {k}')
        if take_step:
            g_alpha, G_alpha = None, None
            if options.mode == 'standard':
                if beta == 0:
                    alpha = 1
                else:
                    b = (1-beta)/(2*eta)
                    c = cfun(k)
                    tau, g_tau, G_tau = binary_search(fg, feature, label, alpha, x, v, L, b, c, f_x_new, ls_tol, options.max_as_iters,
                            options.reduced_tau, options.ls_mode, guess=taufun(k) if options.ls_guess else None,
                            guess_first=options.guess_first)
                    
                    print(tau)
            elif options.mode == 'convex':
                tau = taufun(k)
            elif options.mode == 'agmsdr':
                raise NotImplementedError('AGMSDR not implemented!')
            if tau == 1 or (options.mode == 'convex' and torch.norm(x-v) == 0):
                # latter check is for consistency of our algorithm and regular AGD
                y = x
                f_y = f_x_new
                #df_y = fg(y, sel_feature, sel_label, alpha, grad_only=True) ### add
                df_y = fg(y, feature, label, alpha, grad_only=True)
            else:
                y = tau*x + (1-tau)*v
                if options.mode == 'standard':
                    f_y = g_tau if g_tau is not None else fg(y, feature, label, alpha, func_only=True)
                    #df_y = G_tau if G_tau is not None else fg(y, sel_feature, sel_label, alpha, grad_only=True)
                    df_y = G_tau if G_tau is not None else fg(y, feature, label, alpha, grad_only=True)
                else:
                    f_y = fg(y, feature, label, alpha, func_only=True)
            if options.tol and options.tol_type == 'grad' and torch.max(torch.abs(df_y)) < options.tol:
                fvals.append(f_y)
                k += 1
                break
        
        theta = step_size
        L = 1/theta
        eta = etafun(L, k)
        x_new = y - theta*df_y
        
        v_new = beta*v + (1-beta)*y - eta*df_y
        
        take_step = True
        f_x_new = fg(x_new, feature, label, alpha, func_only=True)
        if options.verbose: print(f'Loss: {f_x_new}')
        if options.tol and options.tol_type == 'func' and f_x_new < options.tol:
            fvals.append(f_x_new)
            k += 1
            break
        
        delta = (f_y - theta*ssq(df_y)/2) - f_x_new #Adaptive stepsize
        if delta >= 0:
            if options.step_type != 'constant':
                step_size *= options.step_inc
        else:
            if delta < -options.num_tol and options.step_type == 'constant':
                raise ValueError('Constant step size too large')
            step_size *= options.step_dec
            take_step = False
            num_nostep += 1
        
        do_restart = (options.restart == 'alpha' and alpha == 1) or \
                     (options.restart == 'grad' and torch.dot(df_y, x-v) / torch.norm(df_y) < 0) or \
                     (options.restart == 'fval' and f_y > f_x)
        take_step = take_step and not do_restart
        
        if take_step:
            x, v = x_new, v_new
            fvals.append(f_x_new)
            k += 1
    func_eval, grad_eval = evals
    return k, num_nostep, func_eval, grad_eval, fvals, x

def agd_strong(fg, gamma, L, mu, x0, options=AGD_options()):
    options.step_type, options.lr = 'constant', 1/L
    beta = 1 - gamma*np.sqrt(mu/L)
    ls_tol = 0
    def etafun(L, k):
        return np.sqrt(1 / (mu*L))
    kappa = np.sqrt(L/mu)
    def cfun(k):
        return kappa
    def alphafun(k):
        return kappa / (1+kappa)
        #return None
    return agd_framework(fg, x0, beta, etafun, cfun, alphafun, ls_tol, options)

def agd_nonstrong(fg, gamma, x0, options=AGD_options()):
    options.reduced_tau = False
    beta = 1
    ls_tol = gamma*options.tol/2
    omegas = [1]
    def omegafun(k):
        assert (len(omegas)-2 <= k <= len(omegas)-1)
        if k == len(omegas) - 1:
            omega = omegas[-1]
            omega = omega/2*(np.sqrt(omega**2+4) - omega)
            omegas.append(omega)
        return omegas[k+1]
    def etafun(L, k):
        Et = gamma*(2*k+3) / (4*L)
        return Et
    def cfun(k):
        seq = (k+1)**2 / (2*k+3)
        return gamma*seq
    def taufun(k):
        omega = omegafun(k)
        return 1 - omega
    return agd_framework(fg, x0, beta, etafun, cfun, taufun, ls_tol, options)

def gd(fg, x0, options=AGD_options()):
    K = options.max_iters
    step_size = options.lr
    evals = [0, 0]
    fg_old = fg
    alpha = options.alpha
    def fg(x, feature, label, alpha, func_only=False, grad_only=False):
        if not grad_only:
            evals[0] += len(feature)
        if not func_only:
            evals[1] += len(feature)
        if grad_only:
            return fg_old(x, feature, label, alpha)[1]
        return fg_old(x, feature, label, alpha, func_only=func_only)

    L = 1 / step_size
    x = x0
    f_x_new = fg(x, feature, label, alpha, func_only=True)
    fvals = [f_x_new]
    print(f_x_new)
    take_step = True
    num_nostep = 0
    k = 0

    while k < K:
        if options.verbose: print(f'Step {k}')
        if take_step:
            f_x = f_x_new
            df_x = fg(x, feature, label, alpha, grad_only=True)
            if options.tol and (options.tol_type == 'grad' and torch.max(torch.abs(df_x)) < options.tol) \
                or (options.tol_type == 'func' and f_x < options.tol):
                fvals.append(f_x)
                k += 1
                break

        theta = step_size
        x_new = x - theta*df_x
        take_step = True
        f_x_new = fg(x_new, feature, label, alpha, func_only=True)
        if options.verbose: print(f'Loss: {f_x_new}')
        delta = (f_x - theta*ssq(df_x)/2) - f_x_new
        if delta >= 0:
            if options.step_type != 'constant':
                step_size *= options.step_inc
        else:
            if delta < -options.num_tol and options.step_type == 'constant':
                raise ValueError('Constant step size too large')
            step_size *= options.step_dec
            take_step = False
            num_nostep += 1
        if take_step:
            x = x_new
            fvals.append(f_x_new)
            k += 1

    func_eval, grad_eval = evals
    return k, num_nostep, func_eval, grad_eval, fvals, x

##Real example
    
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

def Hingeloss(x, feature, label, alpha, func_only=False, grad_only=False):
    #feature should be a matrix and label should be a col vector.
    Efval = 0
    Edfval = torch.zeros(feature.size(dim=1), dtype=feature.dtype)
    for i in range(len(feature)):
        fval, dfval = phi(1-label[i]*torch.dot(feature[i], x), alpha)
        Efval += fval
        Edfval += dfval*label[i]*feature[i]*(-1)
    Efval = Efval / len(feature)
    Edfval = Edfval / len(feature)
    if func_only:
        return Efval
    if grad_only:
        return Edfval
    return Efval, Edfval


def ReHingeloss(x, feature, label, alpha, func_only=False, grad_only=False):
    #feature should be a matrix and label should be a col vector.
    Efval = 0
    Edfval = torch.zeros(feature.size(dim=1), dtype=feature.dtype)
    for i in range(len(feature)):
        fval, dfval = phi(1-label[i]*torch.dot(feature[i], x), alpha)
        Efval += fval
        Edfval += dfval*label[i]*feature[i]*(-1)
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


# Initial guesses - perturbed version of true values
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




options = AGD_options()
gamma = options.alpha
#x0 = torch.tensor([-65.8015, -23.1198, -92.1828, -13.7849])#Initial point of Dataset 1
#k, num_nostep, func_eval, grad_eval, fvals, x = agd_nonstrong(func_wrapper, gamma, params_cat.data, options)
#k, num_nostep, func_eval, grad_eval, fvals, x = agd_nonstrong(Hingeloss, gamma, x0, options=AGD_options())
#k, num_nostep, func_eval, grad_eval, fvals, x = agd_strong(ReHingeloss, gamma, 1, 0.02, x0)
k, num_nostep, func_eval, grad_eval, fvals, x = gd(func_wrapper, params_cat.data, options=AGD_options())
print(func_eval+grad_eval)#output the overall complexity

with open("adapGDLDS5gamma=0.5", "wb") as fp: 
    pickle.dump(fvals, fp)
