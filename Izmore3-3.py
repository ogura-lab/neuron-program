import torch
from torch import distributions, nn, optim
import os
import sys
import random
import numpy as np
import copy
import torch
import torchsde
from multiprocessing import Pool
import datetime
import shutil
import glob
import pandas as pd
import pickle

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import importlib
import pip
spam_spec = importlib.util.find_spec("spam")

import math
from tqdm.notebook import tqdm

# Batch size, number of states, number of time discretization points, and maximum simulation time
batch_size, state_size, t_size, t_max = 100, 2, 400, 20

# Learning rate
lr = 0.25

# Time discretization step size
h = t_max/(t_size-1) 

# Lower and upper bounds of external input
alphamin, alphamax = -100.0, 100.0

# Discretized time points
ts = torch.linspace(0, t_max, t_size)

v0= -30.0
u0 = 0.0

# Index of the element related to firing in the state 
# For the LIF model, it is 0
fireIndex = 0 


class SDE(nn.Module):
  def __init__(self, a, b, c, d):
    super().__init__()

    # Parameter definition
    
    self.beta = torch.tensor(1.0)
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.idealFiringTime = [1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0] # Desired firing times
    self.numFiringTime = len(self.idealFiringTime) # Number of target firings
    self.separateTime = [2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0] # Time points for staged learning
    self.numSeparateTime = len(self.separateTime) + 1 # Number of learning stages
    self.threshold = torch.tensor(30.0) # Membrane potential threshold for firing
    self.gapWeight = torch.tensor(0.1)
    self.gapWeight2 = torch.tensor(0.1)

    self.sList = [0] * self.numSeparateTime
    for i in range(self.numSeparateTime - 1):
      self.sList[i] = math.ceil(self.separateTime[i] / h)
    self.sList[self.numSeparateTime - 1] = t_size

    # Parameters to be learned, converted to alpha using the sigmoid function
    # Initial value is fixed at 0
    self.theta = torch.nn.ParameterList()
    self.theta.append(nn.Parameter(torch.zeros(size=(self.sList[0], 1)), requires_grad=True))
    for i in range(self.numFiringTime-1):
     self.theta.append(nn.Parameter(torch.zeros(size=(self.sList[i+1]-self.sList[i], 1)), requires_grad=False))

    self.noise_type = "diagonal"
    self.sde_type = "ito"
  
  def f(self, t, y): #Compute time derivatives of membrane potential and recovery variable in the neuron model
    v = torch.remainder(y[:, 0]-self.c, self.threshold-self.c)+self.c
    numFire = (y[:, 0]-v)/(self.threshold-self.c) # Compute the number of firings
    u = y[:, 1] + self.d*numFire

    # v' includes a ReLU function
    vprime = torch.tensor(0.04)*v*v + torch.tensor(5.0)*v + torch.tensor(140.0) - u + torch.tensor(5.0)
    
    i = 0
    while t >= self.separateTime[i]:
      i = i + 1
      if i == self.numSeparateTime-1:
        break
    if i == 0:
      vprime = torch.nn.functional.relu(vprime + torch.tensor(alphamin) + torch.tensor(alphamax - alphamin) * torch.tensor(0.5) * (torch.tanh(self.theta[0][math.floor(t/t_max*t_size)]) + torch.tensor(1.0)))
    else:
      vprime = torch.nn.functional.relu(vprime + torch.tensor(alphamin) + torch.tensor(alphamax - alphamin) * torch.tensor(0.5) * (torch.tanh(self.theta[i][math.floor((t-self.separateTime[i-1])/t_max*t_size)]) + torch.tensor(1.0)))
      

    uprime = self.a*(self.b*v-u)
    
    return torch.transpose(torch.stack((vprime, uprime)), 0, 1)
   

  # The part related to noise in the differential equation
  def g(self, t, y):
    return torch.tensor(1.0).repeat(batch_size, 1) * torch.tensor([self.beta, 0])
  
  # Calculate the firing times for the first two spikes
  def firingTime(self, ys, t):
    tArray = torch.zeros(batch_size, t + 2)
    for i in range(batch_size):
      rel_ys = ys[:, i, fireIndex]
      fireFlag = 0
      for j in range(1, self.sList[t]-1):
        if rel_ys[j-1] - self.threshold - fireFlag * (self.threshold - self.c)<0 and rel_ys[j] - self.threshold - fireFlag * (self.threshold - self.c)>0:
          if fireFlag < t + 1:
            # Use linear interpolation to determine the firing time
            tArray[i, fireFlag] = h*(j-1) + h*(self.threshold + fireFlag * (self.threshold - self.c) - rel_ys[j-1])/(rel_ys[j]-rel_ys[j-1])
            fireFlag = fireFlag + 1
          else:
            tArray[i, t + 1] += 1
            fireFlag = fireFlag + 1
    print(tArray)
    return tArray

  # Cost function for training
  def cost(self, ys, t):
    ftArray = self.firingTime(ys, t)
    costArray = torch.zeros(batch_size)
    costArrayex1 = torch.zeros(batch_size)
    costArrayex2 = torch.zeros(batch_size)
    costArrayex3 = torch.zeros(batch_size)
    for i in range(ftArray.size(0)):
      ft = ftArray[i, :]
      if ft[t] == 0: # If the number of firings is less than or equal to the specified number
        if t == 0:#When time t is 0
          voltageGap = self.numFiringTime * torch.sum((self.threshold - ys[:, i, fireIndex])**2)
          costArrayex2[i] = costArrayex2[i] + self.gapWeight * voltageGap
          Iextx = 2 * alphamax - (alphamax - alphamin) * 0.5 * (torch.tanh(self.theta[t][:, 0]) + 1.0)
          costArrayex2[i] = costArrayex2[i] + torch.sum(torch.square(Iextx)) * 1 * self.gapWeight
        else:
          voltageGap = (self.numFiringTime - t) * torch.sum((self.c + (self.threshold - self.c) * (t + 1) - ys[math.ceil(ft[t - 1]/h):, i, fireIndex])**2)
          costArrayex2[i] = costArrayex2[i] + self.gapWeight * voltageGap
          Iextx = 2 * alphamax - (alphamax - alphamin) * 0.5 * (torch.tanh(self.theta[t][:, 0]) + 1.0)
          costArrayex2[i] = costArrayex2[i] + torch.sum(torch.square(Iextx)) * 1 * self.gapWeight
      else: # If a firing has occurred
        costArrayex1[i] = costArrayex1[i] + (ft[t]-self.idealFiringTime[t])**2
      if ft[t + 1] > 0: # If the number of firings exceeds the specified number
        voltageGap = ft[t+1] * torch.sum((ys[math.ceil(ft[t]/h):, i, fireIndex] - self.c - (self.threshold - self.c) * (t+1))**2)
        costArrayex3[i] = costArrayex3[i] + self.gapWeight2 * voltageGap
      costArray[i] = costArrayex1[i] + costArrayex2[i] + costArrayex3[i]
    print(torch.sum(costArrayex1))
    print(torch.sum(costArrayex2))
    print(torch.sum(costArrayex3))
    return costArray

  def printCost(self, ys, x, tstr):#Method to save the cost to a separate file
    ftArray = self.firingTime(ys, self.numFiringTime-1)
    costArray = torch.zeros(batch_size)
    costArrayex1 = torch.zeros(batch_size)
    costArrayex2 = torch.zeros(batch_size)
    costArrayex3 = torch.zeros(batch_size)
    tfile = 'text_' + str(x) + '.txt'
    for i in range(ftArray.size(0)):
      ft = ftArray[i, :]
      for j in range(self.numFiringTime):
        if ft[j] == 0: # If the number of firings is less than or equal to the specified number
          if j == 0:
            voltageGap = self.numFiringTime * torch.sum((self.threshold - ys[:, i, fireIndex])**2)
            costArrayex2[i] = costArrayex2[i] + self.gapWeight * voltageGap
            for t in range(self.numFiringTime):
              Iextx = Iextx + 2 * alphamax - (alphamax - alphamin) * 0.5 * (torch.tanh(self.theta[t][:, 0]) + 1.0)
            costArrayex2[i] = costArrayex2[i] + torch.sum(torch.square(Iextx)) * 1 * self.gapWeight
            break
          else:
            voltageGap = (self.numFiringTime - j) * torch.sum((self.c + (self.threshold - self.c) * (j + 1) - ys[math.ceil(ft[j - 1]/h):, i, fireIndex])**2)
            costArrayex2[i] = costArrayex2[i] + self.gapWeight * voltageGap
            for t in range(self.numFiringTime - j):
              Iextx = Iextx + 2 * alphamax - (alphamax - alphamin) * 0.5 * (torch.tanh(self.theta[t+j][:, 0]) + 1.0)
            costArrayex2[i] = costArrayex2[i] + torch.sum(torch.square(Iextx)) * 1 * self.gapWeight
            break
        else: # If a firing has occurred
          costArrayex1[i] = costArrayex1[i] + (ft[j]-self.idealFiringTime[j])**2
      if ft[self.numFiringTime] > 0: # If the number of firings exceeds the specified number
        voltageGap = ft[self.numFiringTime] * torch.sum((ys[math.ceil(ft[self.numFiringTime-1]/h):, i, fireIndex] - self.c - (self.threshold - self.c) * (self.numFiringTime))**2)
        costArrayex3[i] = costArrayex3[i] + self.gapWeight2 * voltageGap
      costArray[i] = costArrayex1[i] + costArrayex2[i] + costArrayex3[i]
      with open(tfile, mode='a') as f:
        f.write(str(ftArray[i, :]) + '\n' + str(costArray[i]) + '\n')
    print(torch.sum(costArrayex1))
    print(torch.sum(costArrayex2))
    print(torch.sum(costArrayex3))
    with open(tfile, mode='a') as f:
      f.write('\n' + str(torch.sum(costArray)))
    shutil.move('text_' + str(x) + '.txt', 'dir_1_' + tstr)
    return costArray

  def uniplot(self, ts, samples, tstr, xlabel, ylabel, fname, title=''):
    ts = ts.cpu()
    yv = copy.deepcopy(samples[:, :, 0].squeeze().t().cpu())
    numFire = torch.zeros(batch_size, t_size)
    for i in range(yv.size(0)):
      if yv[i, 0] - self.threshold > 0:#If the muscle potential is above the threshold at the initial time, fire once and reset the muscle potential
        numFire[i, 0] = 1
        yv[i, 0] = yv[i, 0] - (self.threshold - self.c)
      for j in range(yv.size(1) - 1):
        numFire[i, j+ 1] = numFire[i, j]
        yv[i, j+ 1] = yv[i, j+ 1] - numFire[i, j+ 1] * (self.threshold - self.c)
        if yv[i, j+ 1] - self.threshold > 0:
          numFire[i, j+ 1] = numFire[i, j+ 1] + 1
          yv[i, j+ 1] = yv[i, j+ 1] - (self.threshold - self.c)
    plt.figure()#Plot only the 0th yv
    plt.plot(ts, yv[0,:])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    plt.savefig(fname)
    shutil.move(fname, 'dir_1_' + tstr)
    plt.show()
    # Save the data
    self.save_data_for_uniplot(ts, samples, tstr, fname)

  def saveplot(self, ts, samples, tstr, fnum, xlabel, ylabel, fname, title=''):
    ts = ts.cpu()
    yv = copy.deepcopy(samples[:, :, 0].squeeze().t().cpu())
    numFire = torch.zeros(batch_size, t_size)
    fire_time = []
    for i in range(yv.size(0)):
      if yv[i, 0] - self.threshold > 0:
        numFire[i, 0] = 1
        yv[i, 0] = yv[i, 0] - (self.threshold - self.c)
      for j in range(yv.size(1) - 1):
        numFire[i, j+ 1] = numFire[i, j]
        yv[i, j+ 1] = yv[i, j+ 1] - numFire[i, j+ 1] * (self.threshold - self.c)
        if yv[i, j+ 1] - self.threshold > 0:
          numFire[i, j+ 1] = numFire[i, j+ 1] + 1
          yv[i, j+ 1] = yv[i, j+ 1] - (self.threshold - self.c)
    plt.figure()#Plot all yv values
    for i, sample in enumerate(yv):
      #Record the points where the value goes from above +10 to below -10
      for k in range(len(sample)):
        if k != 0:
          if sample[k-1]>10 and sample[k]<-10:
            fire_time.append(ts[k-1])
      if i < 10:  # Plot only the first 10 samples
        plt.plot(ts, sample, label=f'sample {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    plt.savefig(fname)
    shutil.move(fname, 'dir_1_' + tstr)
    plt.show()

    fire_time_diff = []
    for j in range(len(fire_time)):
      fire_time_diff.append(-self.idealFiringTime[j%self.numFiringTime]+fire_time[j])
    print(fire_time_diff)
    np_time_diff = np.array(fire_time_diff)
    # Split the data based on the remainder
    mod_0 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 0]
    mod_1 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 1]
    mod_2 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 2]
    mod_3 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 3]
    mod_4 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 4]
    mod_5 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 5]
    mod_6 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 6]
    mod_7 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 7]
    mod_8 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 8]
    mod_9 = np_time_diff[np.arange(len(np_time_diff)) % 10 == 9]
    # Collect the data from each group into a list
    data = {
        'mod_0': mod_0,
        'mod_1': mod_1,
        'mod_2': mod_2,
        'mod_3': mod_3,
        'mod_4': mod_4,
        'mod_5': mod_5,
        'mod_6': mod_6,
        'mod_7': mod_7,
        'mod_8': mod_8,
        'mod_9': mod_9
    }

    df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

    csv_filename = f"time_difference_groups_{fnum}.csv"
    df.to_csv(csv_filename, index=False)
    shutil.move(csv_filename, 'dir_1_' + tstr)
    self.save_data_for_saveplot(ts, samples, tstr, fname)

  def save_data_for_saveplot(self, ts, samples, tstr, fname):
      # Move time steps and sample data to the CPU
      ts = ts.cpu()
      yv = copy.deepcopy(samples[:, :, 0].squeeze().t().cpu())
      numFire = torch.zeros(batch_size, t_size)
      fire_time = []

      # Process the sample data
      for i in range(yv.size(0)):
          if yv[i, 0] - self.threshold > 0:
              numFire[i, 0] = 1
              yv[i, 0] = yv[i, 0] - (self.threshold - self.c)
          for j in range(yv.size(1) - 1):
              numFire[i, j + 1] = numFire[i, j]
              yv[i, j + 1] = yv[i, j + 1] - numFire[i, j + 1] * (self.threshold - self.c)
              if yv[i, j + 1] - self.threshold > 0:
                  numFire[i, j + 1] = numFire[i, j + 1] + 1
                  yv[i, j + 1] = yv[i, j + 1] - (self.threshold - self.c)

      # Record the firing times
      for i, sample in enumerate(yv):
          for k in range(len(sample)):
              if k != 0 and sample[k - 1] > 10 and sample[k] < -10:
                  fire_time.append(ts[k - 1])

      # Create the structure for the data to be saved
      data_to_save = {
          'time_steps': ts.numpy(),
          'processed_samples': yv.numpy(),
          'numFire': numFire.numpy(),
          'fire_time': fire_time,
          'threshold': self.threshold,
          'c_value': self.c,
      }

      # Save the data
      save_path = fname.replace('.pdf', '.pkl')
      with open(save_path, 'wb') as f:
          pickle.dump(data_to_save, f)

      shutil.move(save_path, f'dir_1_{tstr}/saveplot_{save_path}')

      print(f"データを {save_path} に保存しました")

  def save_data_for_uniplot(self, ts, samples, tstr, fname):
      # Move data to the CPU (if necessary)
      ts = ts.cpu()
      yv = copy.deepcopy(samples[:, :, 0].squeeze().t().cpu())
      numFire = torch.zeros(batch_size, t_size)

      # Process the sample data
      for i in range(yv.size(0)):
          if yv[i, 0] - self.threshold > 0:
              numFire[i, 0] = 1
              yv[i, 0] = yv[i, 0] - (self.threshold - self.c)
          for j in range(yv.size(1) - 1):
              numFire[i, j + 1] = numFire[i, j]
              yv[i, j + 1] = yv[i, j + 1] - numFire[i, j + 1] * (self.threshold - self.c)
              if yv[i, j + 1] - self.threshold > 0:
                  numFire[i, j + 1] = numFire[i, j + 1] + 1
                  yv[i, j + 1] = yv[i, j + 1] - (self.threshold - self.c)

      # Create the structure for the data to be saved
      data_to_save = {
          'time_steps': ts.numpy(),
          'first_processed_sample': yv[0, :].numpy(),
          'threshold': self.threshold, 
          'c_value': self.c, 
      }

      # Save the data
      save_path = fname.replace('.pdf', '.pkl')
      with open(save_path, 'wb') as f:
          pickle.dump(data_to_save, f)
      shutil.move(save_path, f'dir_1_{tstr}/uniplot_{save_path}')
      print(f"uniデータを uniplot{save_path} に保存しました") 

def learnIz(a, b, c, d, fnum, tstr):# Function to perform learning
  # Initial values
  y0 = torch.cat((
    torch.full(size=(batch_size, 1), fill_value=v0),
    torch.full(size=(batch_size, 1), fill_value=u0)), 1)
    
  seed = 0

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  sde = SDE(a,b,c,d)
  with torch.no_grad():
      ys = torchsde.sdeint(sde, y0, ts, method='euler')  # (t_size, batch_size, state_size) 
  
  # Display before training
  sde.uniplot(ts, ys, tstr, xlabel='$t$', ylabel='$Y_t$', title='membrane potential', fname="zeropotential_" + str(fnum) + ".pdf")
  plt.figure()
  tmpTheta = copy.deepcopy(sde.theta[0])
  for i in range(sde.numFiringTime-1):
    tmpTheta2 = copy.deepcopy(sde.theta[i+1])
    tmpTheta = torch.cat((tmpTheta, tmpTheta2), 0)
  Iext = alphamin + (alphamax - alphamin) * 0.5 * (torch.tanh(tmpTheta[0:t_size, :]) + 1.0).detach().numpy()
  plt.plot(ts, Iext)
  plt.title('input')
  plt.xlabel('$t$')
  plt.ylabel('$Iext(t)$')
  plt.savefig("zeroinput_" + str(fnum) + ".pdf")
  shutil.move("zeroinput_" + str(fnum) + ".pdf", 'dir_1_' + tstr)
  plt.show()

  loss = sde.cost(ys, sde.numFiringTime - 1).sum(0) 

  # Display cost function value
  print(loss)

  optimizer = optim.Adam(sde.parameters(), betas = (0.2, 0.36), lr=lr) # Specify optimizer
  epochNum = 500 #Number of training iterations

  n = 0
  aLoss = torch.zeros(epochNum * sde.numFiringTime).detach().numpy()

  for l in range(sde.numFiringTime):
    if l > 0:
      optimizer.zero_grad() 
      sde.theta[l-1].requires_grad_(False)
      sde.theta[l].requires_grad_(True)
      sde.theta[l-1] = copy.deepcopy(sde.theta[l-1]) # Prevent theta from being updated
      m = 0
    for k in tqdm(range(epochNum)):
      print(n + 1)
        
      # Pre-update parameter process
      optimizer.zero_grad() 

      # Compute cost function
      loss = sde.cost(torchsde.sdeint(sde, y0, ts, method='euler'), l).sum(0)
      # Display cost function value
      print(loss) 

      n = n + 1

      # Compute gradient in the direction that minimizes the cost function and update parameters
      loss.backward()
      
      print("Iext:" + str(fnum))
      tmpTheta = copy.deepcopy(sde.theta[0])
      for i in range(sde.numFiringTime-1):
        tmpTheta2 = copy.deepcopy(sde.theta[i+1])
        tmpTheta = torch.cat((tmpTheta, tmpTheta2), 0)
      Iext = alphamin + (alphamax - alphamin) * 0.5 * (torch.tanh(tmpTheta[0:t_size, :]) + 1.0).detach().numpy()
      print(Iext.squeeze())
      print("gradient:" + str(fnum))
      print(sde.theta[l].grad.squeeze())
      #Save intermediate input progress, currently saving 4 times per firing
      if k == 0 or k == int(epochNum*1/4) or k == int(epochNum/2) or k==int(epochNum*3/4):
        result = torch.cat([sde.theta[i] for i in range(sde.numFiringTime)], dim=0)
        plt.plot(ts, result.detach().numpy())
        plt.title('input')
        plt.xlabel('$t$')
        plt.ylabel('$Iext(t)$')
        plt.savefig("input_" + str(fnum) + "_" + str(l) + "_" + str(k) + ".pdf")
        shutil.move("input_" + str(fnum) + "_" + str(l) + "_" + str(k) + ".pdf", 'dir_1_' + tstr)
        plt.close()
      
      if k == 0 or k == int(epochNum * 1/4) or k == int(epochNum / 2) or k == int(epochNum * 3/4):
        # Code for saving data
        data_to_save = {
            'ts': ts.numpy(),  # Time steps
            'theta': [sde.theta[i].detach().numpy() for i in range(sde.numFiringTime)],  # All sde.theta[i] data
            'fnum': fnum,
            'l': l,
            'epochNum': epochNum,
        }

        # Save directory
        save_dir = f'dir_1_{tstr}'

        # Save data
        save_path = f"{save_dir}/input_data_{fnum}_{l}_mid_input_data_{fnum}_{l}_epoch_{k}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"データを {save_path} に保存しました")

      # Display parameters before updating
      optimizer.step() 

  # Changes in membrane potential
  y0 = torch.cat((
      torch.full(size=(batch_size, 1), fill_value=v0),
      torch.full(size=(batch_size, 1), fill_value=u0)), 1)
  with torch.no_grad():
      ys = torchsde.sdeint(sde, y0, ts, method='euler')
  sde.saveplot(ts, ys, tstr, fnum, xlabel='$t$', ylabel='$Y_t$', title='membrane potential', fname="potential_" + str(fnum) + ".pdf")
  sde.uniplot(ts, ys, tstr, xlabel='$t$', ylabel='$Y_t$', title='membrane potential', fname="unipotential_" + str(fnum) + ".pdf")

  plt.figure()
  tmpTheta = copy.deepcopy(sde.theta[0])
  for i in range(sde.numFiringTime-1):
    tmpTheta2 = copy.deepcopy(sde.theta[i+1])
    tmpTheta = torch.cat((tmpTheta, tmpTheta2), 0)
  Iext = alphamin + (alphamax - alphamin) * 0.5 * (torch.tanh(tmpTheta[0:t_size, :]) + 1.0).detach().numpy()
  plt.plot(ts, Iext)
  plt.title('input')
  plt.xlabel('$t$')
  plt.ylabel('$Iext(t)$')
  plt.savefig("input_" + str(fnum) + ".pdf")
  shutil.move("input_" + str(fnum) + ".pdf", 'dir_1_' + tstr)
  plt.show()
  plt.close()

  # Save time steps (ts) and input data (Iext)
  data_to_save = {
      'ts': ts.numpy(), 
      'Iext': Iext,
      'fnum': fnum,
  }
  save_dir = f'dir_1_{tstr}'  # Save directory

  # Save data
  save_path = f"{save_dir}/input_data_{fnum}.pkl"
  with open(save_path, 'wb') as f:
      pickle.dump(data_to_save, f)
  print(f"データを {save_path} に保存しました")

  # Firing time
  ft = sde.firingTime(ys, sde.numFiringTime - 1)
  print(ft.detach().numpy())

  costArray = torch.zeros(batch_size)
  for i in range(batch_size):
    costArray[i] = (ft[i,0]-sde.idealFiringTime[0])**2 + (ft[i,1]-sde.idealFiringTime[1])**2
  print(costArray[0])
  print(costArray.sum(0))

  sde.printCost(torchsde.sdeint(sde, y0, ts, method='euler'),fnum, tstr)

def learnIz_wrapper(args):
  return learnIz(*args)

def returnValues(n, tstr):
  if n == 1:
    values = [[0.02, 0.2,-65.0,8.0,1,tstr],[0.02, 0.2,-55.0,4.0,2,tstr],[0.02,0.2,-50.0,2.0,3,tstr],[0.1,0.2,-65.0,2.0,4,tstr],[0.02,0.25,-65.0,0.05,5,tstr],[0.1,0.26,-65.0,2.0,6,tstr],[0.02,0.25,-65.0,2.0,7,tstr]]
  elif n == 2:
    values = [[0.02, 0.2,-55.0,4.0,2,tstr]]
  return values


def timestr():
    MYTIMEZONE = 'JST'
    JST = datetime.timezone(datetime.timedelta(hours=+9), MYTIMEZONE)
    now = datetime.datetime.now(JST)
    return '{0:%Y%m%d-%H%M%S}'.format(now)

if __name__ == "__main__":
    p = Pool(7)
    n = 1
    m = 1
    tstr = timestr()
    os.mkdir('dir_' + str(m) + '_' + tstr)
    shutil.copy2("Izmore3-3.py", "dir_" + str(m) + "_" + tstr + "/")

    p.map(learnIz_wrapper, returnValues(n, tstr))
    for a in glob.glob('*.pdf', recursive=True):
      shutil.move(a, 'dir_' + str(m) + '_' + tstr)
    for a in glob.glob('*.txt', recursive=True):
      shutil.move(a, 'dir_' + str(m) + '_' + tstr)