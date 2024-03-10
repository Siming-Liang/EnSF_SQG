import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import sys
import os

class EnSF:
    def __init__(self, n_dim, ensemble_size,eps_alpha, device,obs_sigma,euler_steps,scalefact,init_std_x_state, ISarctan=False ):
    ####################################################################
    # EnSF setup
    # define the diffusion process eps_alpha
    # ensemble size ensemble_size = 250
        self.n_dim = n_dim
        self.ensemble_size = ensemble_size
        self.eps_alpha = eps_alpha
        self.device = device
        self.obs_sigma = obs_sigma
        self.ISarctan = ISarctan
        self.euler_steps = euler_steps
        self.scalefact = scalefact
        self.init_std_x_state = init_std_x_state
    # computation setting
        #torch.set_default_dtype(torch.float16) # half precision
        #device = 'cuda' 'cpu'

# compact version
    def cond_alpha(self,t):
        # conditional information
        # alpha_t(0) = 1
        # alpha_t(1) = esp_alpha \approx 0
        return 1 - (1-self.eps_alpha)*t

    def cond_sigma_sq(self,t):
    # conditional sigma^2
    # sigma2_t(0) = 0
    # sigma2_t(1) = 1
    # sigma(t) = t
        return t

# drift function of forward SDE
    def f(self,t):
        # f=d_(log_alpha)/dt
        alpha_t = self.cond_alpha(t)
        f_t = -(1-self.eps_alpha) / alpha_t
        return f_t


    def g_sq(self,t):
        # g = d(sigma_t^2)/dt -2f sigma_t^2
        d_sigma_sq_dt = 1
        g2 = d_sigma_sq_dt - 2*self.f(t)*self.cond_sigma_sq(t)
        return g2

    def g(self,t):
        return np.sqrt(self.g_sq(t))


# generate sample with reverse SDE
    def reverse_SDE(self, obs,x0, time_steps,obs_sigma,sparse_idx, arc_idx,save_path=False):
        # x_T: sample from standard Gaussian
        # x_0: target distribution to sample from
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device
        #drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,  score_likelihood=None, 
        # reverse SDE sampling process
        # N1 = x_T.shape[0]
        # N2 = x0.shape[0]
        # d = x_T.shape[1]

        # Generate the time mesh
        dt = 1.0/time_steps

        # Initialization
        xt = torch.randn(ensemble_size,n_dim, device=device)
        t = 1.0

        # define storage
        if save_path:
            path_all = [xt]
            t_vec = [t]
        mart_point = int(0.20 * time_steps)  ###0.20
        temp_state = torch.zeros(mart_point,n_dim)
        # forward Euler sampling
        for i in range(time_steps):
            temp_mean_old = temp_state.mean(dim = 0)
            # prior score evaluation
            alpha_t = self.cond_alpha(t)#alpha_fun(t)
            sigma2_t = self.cond_sigma_sq(t)#sigma2_fun(t)

            # Evaluate the diffusion term
            diffuse = self.g(t) #diffuse_fun(t)

            # Update
            xt += - dt*( self.f(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) - diffuse**2 * self.score_likelihood(xt, t,obs,obs_sigma,sparse_idx,arc_idx) ) \
                    + np.sqrt(dt)*diffuse*torch.randn_like(xt)

            # Store the state in the path
            if save_path:
                path_all.append(xt)
                t_vec.append(t)
            
            #save moving state
            temp_state[i % mart_point,:] = xt.mean(dim = 0)
            if (abs(temp_state.mean(dim = 0) - temp_mean_old) < 0.005).all() and i >mart_point: 
                break
            else:
                pass
            if i > 500:
                break
            # update time
            t = t - dt

        if save_path:
            return path_all, t_vec
        else:
            return xt

    # damping function(tau(0) = 1;  tau(1) = 0;)
    def g_tau(self, t):
        return 1-t

# define likelihood score
    def score_likelihood(self, xt, t,obs,obs_sigma,sparse_idx,arc_idx):
        # obs: (d)
        # xt: (ensemble, d)
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device        
        score_x = torch.zeros(ensemble_size,n_dim, device=device)       
        score_x[:,arc_idx] = (-(torch.atan(self.scalefact*xt[:,arc_idx]) - obs[arc_idx])/(obs_sigma[arc_idx])**2 * (self.scalefact*1./(1. + self.scalefact**2 * (xt[:,arc_idx])**2))).type_as(score_x)
        score_x[:,~arc_idx] = (-( self.scalefact*xt[:,~arc_idx] - obs[~arc_idx])/obs_sigma[~arc_idx]**2 * self.scalefact ).type_as(score_x)
        score_x[:,~sparse_idx] = 0 
        tau = self.g_tau(t)

        return tau*score_x



    def state_update(self,x_input,state_target_input, obs_input):
        # filtering settings
        ensemble_size = self.ensemble_size
        # observation sigma
        # forward Euler step
        euler_steps = self.euler_steps
        #####################################################################################################################    
        # filtering ensemble
        x_state = torch.tensor(x_input,device='cuda')
        state_target = torch.tensor(state_target_input,device='cuda')
        # get observation
        obs = torch.tensor(obs_input,device='cuda')
        temp_obs_sigma = self.obs_sigma * self.scalefact 

        '''
        # get state memory size
        mem_state = x_state.element_size() * x_state.nelement()/1e+6
        mem_ensemble = mem_state * ensemble_size
        print(f'single state memory: {mem_state:.2f} MB')
        print(f'state ensemble memory: {mem_ensemble:.2f} MB')
        '''
        torch.cuda.empty_cache()

        t1 = time.time()

        # generate posterior sample
        x_state = self.reverse_SDE(obs=obs,x0=x_state,time_steps=euler_steps,obs_sigma = temp_obs_sigma) #self.obs_sigma
        # get state estimates
        x_est = torch.mean(x_state,dim=0)

        # get rmse
        rmse_temp = torch.sqrt(torch.mean((x_est - state_target)**2)).item()

        # get time
        if x_state.device.type == 'cuda':
            torch.cuda.current_stream().synchronize()

        t2 = time.time()
        print(f'\t RMSE = {rmse_temp:.4f}')
        #print(f'\t time = {t2-t1:.4f} ')

        # Diverge Warning
        #if rmse_temp > 1000:
        #    print('diverge!')

        return  x_state#, x_est, rmse_temp 
    

    def state_update_normalized(self,x_input,state_target_input, obs_input,sparse_idx,current,arcindex):
        torch.set_default_dtype(torch.float32)
        # filtering settings
        ensemble_size = self.ensemble_size
        # observation sigma
        # forward Euler step
        euler_steps = self.euler_steps
        #####################################################################################################################    
        # filtering ensemble
        x_state = torch.tensor(x_input,device='cuda')
        state_target = torch.tensor(state_target_input,device='cuda')
        init_std_x_state = torch.tensor(self.init_std_x_state,device='cuda')

        # get observation/index
        sparse_index = torch.zeros(x_state.size(dim = 1),dtype=torch.bool,device='cuda')
        sparse_index[sparse_idx] = True

        obs = torch.zeros(x_state.size(dim = 1),device='cuda')
        obs[sparse_index] = torch.tensor(obs_input,device='cuda').type_as(obs)
        arc_index = torch.zeros(x_state.size(dim = 1),dtype=torch.bool,device='cuda')
        arc_index[arcindex] = True
   
   
        #normalize
        std_x_state = x_state.std(dim=0)
        mean_x_state = x_state.mean(dim=0)
        x_state = (x_state- mean_x_state) / std_x_state
        obs[~arc_index] = (((obs[~arc_index] / self.scalefact - mean_x_state[~arc_index]) / std_x_state[~arc_index]) * self.scalefact).type_as(obs) 
        obs[arc_index] = (torch.atan((torch.tan(obs[arc_index]) / self.scalefact - mean_x_state[arc_index])* self.scalefact/ std_x_state[arc_index] )).type_as(obs)   

        temp_obs_sigma = torch.zeros(x_state.size(dim = 1), device='cuda')
        temp_obs_sigma[~arc_index] = (self.obs_sigma  / std_x_state[~arc_index]).type_as(temp_obs_sigma)  
        temp_obs_sigma[arc_index] = (0.01 / std_x_state[arc_index] * 50).type_as(temp_obs_sigma)


        
        '''
        # get state memory size
        mem_state = x_state.element_size() * x_state.nelement()/1e+6
        mem_ensemble = mem_state * ensemble_size
        print(f'single state memory: {mem_state:.2f} MB')
        print(f'state ensemble memory: {mem_ensemble:.2f} MB')
        '''
        torch.cuda.empty_cache()

        t1 = time.time()

        # generate posterior sample
        x_state = self.reverse_SDE(obs=obs,x0=x_state,time_steps=euler_steps,obs_sigma = temp_obs_sigma,sparse_idx = sparse_index,arc_idx = arc_index) 
        x_state = x_state * std_x_state  + mean_x_state 
        x_out = x_state
        # get state estimates
        x_est = torch.mean(x_state,dim=0)
        
        # get rmse
        rmse_temp = torch.sqrt(torch.mean((x_est - state_target)**2)).item()
       
        #if current> 10 and rmse_temp > 350: #current> 10 and
        x_state_temp = (x_state - x_state.mean(dim=0)) * (std_x_state/x_state.std(dim=0) )  + x_state.mean(dim=0)
        if (x_state_temp.std() > (1.5 * init_std_x_state)).any():
            x_state = (x_state - x_state.mean(dim=0)) * (init_std_x_state/x_state.std(dim=0) )  + x_state.mean(dim=0)
        else:
            x_state =  x_state_temp

        # get time
        if x_state.device.type == 'cuda':
            torch.cuda.current_stream().synchronize()

        t2 = time.time()
        #print(f'\t RMSE = {rmse_temp:.4f}')
        #print(f'\t time = {t2-t1:.4f} ')

        # Diverge Warning
        #if rmse_temp > 1000:
        #    print('diverge!')

        return  x_state #x

    
