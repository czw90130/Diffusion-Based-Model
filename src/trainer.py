import os
from functools import partial

from .network import UNetModel,EMA
from .dataloader import load_data, gcloth_mask_uv
from .diffusion import GaussianDiffusion,extract
import .ops

import torch
import torch.optim as optim
import torch.nn as nn

#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import copy

# 配置信息 | 默认值 | 备注
# MODE | 1 | 1 Train, 2 Validation
# IMAGE_SIZE | [224,224] |
# CHANNEL_X | 1
# CHANNEL_Y | 3
# TIMESTEPS | 2000
# MODEL_CHANNELS | 128 | base channel count for the model.
# NUM_RESBLOCKS | 4 | D
# ATTENTION_RESOLUTIONS | [2,4,8] | a collection of downsample rates at which attention will take place. May be a set, list, or tuple. For example, if this contains 4, then at 4x downsampling, attention will be used.
# DROPOUT | 0 | the dropout probability.
# CHANNEL_MULT | [1,2,4,8】 | channel multiplier for each level of the UNet.
# CONV_RESAMPLE | True | if True, use learned convolutions for upsampling and downsampling.
# USE_CHECKPOINT | False | use gradient checkpointing to reduce memory usage.
# USE_FP16 | False | 
# NUM_HEADS | 1 | the number of attention heads in each attention layer.
# NUM_HEAD_CHANNELS | 64 | if specified, ignore num_heads and instead use a fixed channel width per attention head.
# NUM_HEAD_UPSAMPLE | -1 | works with num_heads to set a different number of heads for upsampling. Deprecated.
# USE_SCALE_SHIFT_NORM | False | use a FiLM-like conditioning mechanism.
# RESBLOCK_UPDOWN | False | use residual blocks for up/downsampling.
# USE_NEW_ATTENTION_ORDER | False | use a different attention pattern for potentially increased efficiency.
# PATH_COLOR
# PATH_GREY
# BATCH_SIZE | 1 |
# BATCH_SIZE_VAL | 8 |
# ITERATION_MAX | 1000000 |
# LR | 0.0001 | Adam lr
# LOSS | 'L2' |
# VALIDATION_EVERY | 1000 |
# EMA_EVERY | 100 |
# START_EMA | 2000 |
# SAVE_MODEL_EVERY | 10000 | 

class Trainer():
    def __init__(self,config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.diffusion = GaussianDiffusion(config.IMAGE_SIZE,config.CHANNEL_X,config.CHANNEL_Y,config.TIMESTEPS)

        # 模型加载
        train_list, valid_list = load_data(config.INDEX_FILE, True)
        dataset_train = gcloth_mask_uv(train_list, config.IMAGE_SIZE[0], config.LOW_THRES, config.UP_THRES)  
        self.dataloader_train = DataLoader(dataset_train,batch_size=self.batch_size, shuffle=True)
        dataset_validation = gcloth_mask_uv(valid_list, config.IMAGE_SIZE[0], config.LOW_THRES, config.UP_THRES) 
        self.dataloader_validation = DataLoader(dataset_validation,batch_size=self.batch_size_val,shuffle=False)

        self.network = UNetModel(
            config.IMAGE_SIZE,
            dataset_train.IN_CHANNELS, # channels in the input Tensor, for image colorization : Y_channels + X_channels .
            config.MODEL_CHANNELS, # base channel count for the model.
            dataset_train.OUT_CHANNELS, # channels in the output Tensor.
            config.NUM_RESBLOCKS, # D
            config.ATTENTION_RESOLUTIONS, # a collection of downsample rates at which attention will take place. May be a set, list, or tuple. For example, if this contains 4, then at 4x downsampling, attention will be used.
            config.DROPOUT, # the dropout probability.
            config.CHANNEL_MULT, # channel multiplier for each level of the UNet.
            config.CONV_RESAMPLE, # if True, use learned convolutions for upsampling and downsampling.
            config.USE_CHECKPOINT, # use gradient checkpointing to reduce memory usage.
            config.USE_FP16,
            config.NUM_HEADS, # the number of attention heads in each attention layer.
            config.NUM_HEAD_CHANNELS, # if specified, ignore num_heads and instead use a fixed channel width per attention head.
            config.NUM_HEAD_UPSAMPLE, # works with num_heads to set a different number of heads for upsampling. Deprecated.
            config.USE_SCALE_SHIFT_NORM, # use a FiLM-like conditioning mechanism.
            config.RESBLOCK_UPDOWN, # use residual blocks for up/downsampling.
            config.USE_NEW_ATTENTION_ORDER, # use a different attention pattern for potentially increased efficiency.
            ).to(self.device)
        
        
        self.batch_size = config.BATCH_SIZE
        self.batch_size_val = config.BATCH_SIZE_VAL

        self.iteration_max = config.ITERATION_MAX
        self.EMA = EMA(0.9999)
        self.LR = config.LR
        if config.LOSS == 'L1':
            self.loss = nn.L1Loss()
        if config.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else :
            print('Loss not implemented, setting the loss to L2 (default one)')
        self.num_timesteps = config.TIMESTEPS
        self.validation_every = config.VALIDATION_EVERY
        self.ema_every = config.EMA_EVERY
        self.start_ema = config.START_EMA
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.ema_model = copy.deepcopy(self.network).to(self.device)

    def save_model(self,name,EMA=False):
        if not EMA:
            torch.save(self.network.state_dict(),name)
        else:
            torch.save(self.ema_model.state_dict(),name)

    def train(self):

            to_torch = partial(torch.tensor, dtype=torch.float32)
            optimizer = optim.Adam(self.network.parameters(),lr=self.LR)
            iteration = 0
            
            print('Starting Training')

            while iteration < self.iteration_max:

                tq = tqdm(self.dataloader_train)
                
                for grey,color in tq:
                    tq.set_description(f'Iteration {iteration} / {self.iteration_max}')
                    self.network.train()
                    optimizer.zero_grad()

                    t = torch.randint(0, self.num_timesteps, (self.batch_size,)).long()
                    noisy_image,noise_ref = self.diffusion.noisy_image(t,color)
                    noise_pred = self.diffusion.noise_prediction(self.network,noisy_image.to(self.device),grey.to(self.device),t.to(self.device))
                    loss = self.loss(noise_ref.to(self.device),noise_pred)
                    loss.backward()
                    optimizer.step()
                    tq.set_postfix(loss = loss.item())
                    
                    iteration+=1

                    if iteration%self.ema_every == 0 and iteration>self.start_ema:
                        print('EMA update')
                        self.EMA.update_model_average(self.ema_model,self.network)

                    if iteration%self.save_model_every == 0:
                        print('Saving models')
                        if not os.path.exists('models/'):
                            os.makedirs('models')
                        self.save_model(f'models/model_{iteration}.pth')
                        self.save_model(f'models/model_ema_{iteration}.pth',EMA=True)

                    if iteration%self.validation_every == 0:
                        tq_val = tqdm(self.dataloader_validation)
                        with torch.no_grad():
                            self.network.eval()
                            for grey,color in tq_val:
                                tq_val.set_description(f'Iteration {iteration} / {self.iteration_max}')
                                T = 1000
                                alphas = np.linspace(1e-4,0.09,T)
                                gammas = np.cumprod(alphas,axis=0)
                                y = torch.randn_like(color)
                                for t in range(T):
                                    if t == 0 :
                                        z = torch.randn_like(color)
                                    else:
                                        z = torch.zeros_like(color)

                                    time = (torch.ones((self.batch_size_val,)) * t).long()
                                    y = extract(to_torch(np.sqrt(1/alphas)),time,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),time,y.shape))*self.network(y.to(self.device),grey.to(self.device),time.to(self.device)).detach().cpu()) + extract(to_torch(np.sqrt(1-alphas)),time,z.shape)*z
                                    y_ema = extract(to_torch(np.sqrt(1/alphas)),time,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),time,y.shape))*self.ema_model(y.to(self.device),grey.to(self.device),time.to(self.device)).detach().cpu()) + extract(to_torch(np.sqrt(1-alphas)),time,z.shape)*z
                                loss = self.loss(color,y)
                                loss_ema = self.loss(color,y_ema)
                                tq_val.set_postfix({'loss' : loss.item(),'loss ema':loss_ema.item()})
                    

            

        