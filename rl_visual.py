import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
import os

date = '2021-09-23'
time = '13:02:39'
pth=f'/scratch/kw2815/RL-Finance/multirun/{date}/{time}'
save_pth = f'/scratch/kw2815/RL-Finance/visuals/tmp/{date}/{time}'

gamma=[0.99,0.8,0.85] 
gae_lambda=[0.6,0.65,0.7] 
net_arch=['[32,64,128,128,64]','[[64,64,256,256,64]']

if not os.path.isdir(save_pth):
    folder_path = f'/scratch/kw2815/RL-Finance/visuals/{date}'
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    os.mkdir(save_pth)
    os.mkdir(save_pth+'/losses')
    os.mkdir(save_pth+'/rewards')



def plot_loss(lam, gamma, net_arch, loss1, loss2, pth,save_pth, df=None,cutoff=0.1):
    try:
        if not isinstance(df,pd.core.frame.DataFrame):
            folder_pth = pth+f'/gae_lambda={lam},gamma={gamma},net_arch={net_arch}'
            df = pd.read_json(folder_pth+'/rl_logs.json',lines=True).fillna(0)
        else:
            df = df
        fig, ax1 = plt.subplots(figsize=[10,5])

        ax1.set_xlabel('time')
        ax1.set_ylabel(loss1)
        plt1_data = df[f'train/{loss1}_loss']
        
        lpb1 = np.quantile(plt1_data,cutoff)
        upb1 = np.quantile(plt1_data,1-cutoff)    
        np.clip(plt1_data,lpb1, upb1, inplace=True)
        
        ax1.plot(np.arange(len(plt1_data)), plt1_data,label=loss1,c='g')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()

        ax2.set_ylabel(loss2)
        plt2_data = df[f'train/{loss2}_loss']

        lpb2 = np.quantile(plt2_data,cutoff)
        upb2 = np.quantile(plt2_data,1-cutoff)    
        np.clip(plt2_data,lpb2, upb2, inplace=True)
        
        ax2.plot(np.arange(len(plt2_data)), plt2_data,label=loss2,c='r')
        ax2.tick_params(axis='y')
        title = r'$\lambda=$'+str(lam)+', '+'$\gamma=$'+str(gamma)+', net='+str(net_arch)
        plt.title(title)
        fig.legend()
        
        fig.savefig(save_pth+'/losses'+f'/gae_lambda={lam},gamma={gamma},net_arch={net_arch}'+'.png')
    except Exception as e:
        print(e)
def plot_reward(lam, gamma,net_arch,loss1, loss2, save_pth,pth=None,df=None,freq=1000):
    try:
        if not isinstance(df,pd.core.frame.DataFrame):
            folder_pth = pth+f'/gae_lambda={lam},gamma={gamma},net_arch={net_arch}'
            df = pd.read_hdf(folder_pth+'/reward_history.h5')
        else:    
            df = df        
            
        plot_data = df.groupby('episode_num').last()[['revenue','curr_wealth', 'reward']].iloc[::freq,:]
        del df
        
        fig, ax1 = plt.subplots(figsize=[10,5])

        ax1.set_xlabel('time')
        ax1.set_ylabel(loss1)
        ax1.plot(np.arange(len(plot_data)), plot_data[f'{loss1}'],label=loss1,c='g')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()

        ax2.set_ylabel(loss2)
        ax2.plot(np.arange(len(plot_data)), plot_data[f'{loss2}'],label=loss2,c='r')
        ax2.tick_params(axis='y')
        title = r'$\lambda=$'+str(lam)+', '+'$\gamma=$'+str(gamma)+', net='+str(net_arch)
        plt.title(title)
        fig.legend()
        
        fig.savefig(save_pth+'/rewards'+f'/gae_lambda={lam},gamma={gamma},net_arch={net_arch}'+'.png')
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    for ga, gae, net in itertools.product(gamma, gae_lambda, net_arch):
        plot_loss(gae, ga, net,'entropy', 'value',pth, save_pth,cutoff=0.05)
        plot_loss(gae, ga, net,'entropy', 'value',pth,save_pth,cutoff=0.05)
        plot_reward(gae, ga, net, 'revenue', 'reward', save_pth, pth=pth,freq=1000)
    
    