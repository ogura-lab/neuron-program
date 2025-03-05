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

# バッチサイズ，状態数，時間方向の離散化の数，シミュレーションの最大時刻　　#batch_sizeは結果のpotentialの本数
batch_size, state_size, t_size, t_max = 100, 2, 400, 20

# 学習率
lr = 0.25

# 時間方向の離散化の幅
h = t_max/(t_size-1) 

# 外部入力の下限と上限
alphamin, alphamax = -100.0, 100.0

# 離散化した時刻
ts = torch.linspace(0, t_max, t_size)   #torch.linspace(0, 10, 5)=tensor([0.0, 2.5, 5.0, 7.5, 10.0])

v0= -30.0
u0 = 0.0



# 状態において発火にかかわる要素の index．LIFモデルの場合は 0．
fireIndex = 0 

# bm = torchsde.BrownianInterval(t0=0.0, t1=t_max, size=(batch_size, state_size)) # ノイズの時系列

class SDE(nn.Module):
  def __init__(self, a, b, c, d):
    super().__init__()

    # パラメータ定義
    
    self.beta = torch.tensor(1.0) #これがノイズの大きさ
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.idealFiringTime = [1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0] # 発火させたい時刻
    self.numFiringTime = len(self.idealFiringTime) # 目的の発火回数
    self.separateTime = [2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0] # 学習を段階分けする時刻
    self.numSeparateTime = len(self.separateTime) + 1 # 学習段階の数
    self.threshold = torch.tensor(30.0) # 発火する際の膜電位
    self.gapWeight = torch.tensor(0.1)
    self.gapWeight2 = torch.tensor(0.1)

    self.sList = [0] * self.numSeparateTime
    for i in range(self.numSeparateTime - 1):
      self.sList[i] = math.ceil(self.separateTime[i] / h)
    self.sList[self.numSeparateTime - 1] = t_size

    # 学習するパラメータ．sigmoid関数でalphaに変換する
    # 初期値0固定
    self.theta = torch.nn.ParameterList()
    self.theta.append(nn.Parameter(torch.zeros(size=(self.sList[0], 1)), requires_grad=True))
    for i in range(self.numFiringTime-1):
     self.theta.append(nn.Parameter(torch.zeros(size=(self.sList[i+1]-self.sList[i], 1)), requires_grad=False))
    # self.theta = nn.Parameter(torch.zeros(size=(self.sList[0], 1)), requires_grad=True)
    # self.theta2 = nn.Parameter(torch.zeros(size=(t_size-self.sList[0]+1, 1)), requires_grad=False) 
    # 初期値ランダム
    # self.theta = nn.Parameter(torch.normal(0, 1, size=(t_size+1, 1)), requires_grad=True) 

    # おまじない
    self.noise_type = "diagonal"
    self.sde_type = "ito"
  
  def f(self, t, y): #神経細胞モデルの膜電位と回復変数の時間微分を計算する
    # y[:, 0]: リセットなしのv
    # y[:, 1]: ジャンプなしのu
    v = torch.remainder(y[:, 0]-self.c, self.threshold-self.c)+self.c
    numFire = (y[:, 0]-v)/(self.threshold-self.c) # 発火回数を計算
    u = y[:, 1] + self.d*numFire

    # v' は relu 関数をいれてズルをしている
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
      
    # if t < self.separateTime[0]:
    #   vprime = torch.nn.functional.relu(vprime + torch.tensor(alphamin) + torch.tensor(alphamax - alphamin) * torch.tensor(0.5) * (torch.tanh(self.theta[math.floor(t/t_max*t_size)]) + torch.tensor(1.0)))
    # else:
    #   vprime = torch.nn.functional.relu(vprime + torch.tensor(alphamin) + torch.tensor(alphamax - alphamin) * torch.tensor(0.5) * (torch.tanh(self.theta2[math.floor((t-self.separateTime[0])/t_max*t_size)]) + torch.tensor(1.0)))

    uprime = self.a*(self.b*v-u)
    
    return torch.transpose(torch.stack((vprime, uprime)), 0, 1)
   

  # 微分方程式においてノイズに関係する部分
  # ここは暫定のコードになっている
  def g(self, t, y):
    return torch.tensor(1.0).repeat(batch_size, 1) * torch.tensor([self.beta, 0])
    # return self.beta.repeat(batch_size, state_size) 
  
  # 2回目までの発火時間を計算する
  def firingTime(self, ys, t):
    tArray = torch.zeros(batch_size, t + 2)
    for i in range(batch_size):
      rel_ys = ys[:, i, fireIndex]
      fireFlag = 0
      for j in range(1, self.sList[t]-1):
        if rel_ys[j-1] - self.threshold - fireFlag * (self.threshold - self.c)<0 and rel_ys[j] - self.threshold - fireFlag * (self.threshold - self.c)>0:
          if fireFlag < t + 1:
            # 線形補間をつかって発火時間を決定
            tArray[i, fireFlag] = h*(j-1) + h*(self.threshold + fireFlag * (self.threshold - self.c) - rel_ys[j-1])/(rel_ys[j]-rel_ys[j-1])
            fireFlag = fireFlag + 1
          else:
            tArray[i, t + 1] += 1
            fireFlag = fireFlag + 1
    print(tArray)
    return tArray

  # 学習に用いるコスト関数
  def cost(self, ys, t):
    ftArray = self.firingTime(ys, t)
    costArray = torch.zeros(batch_size)
    costArrayex1 = torch.zeros(batch_size)
    costArrayex2 = torch.zeros(batch_size)
    costArrayex3 = torch.zeros(batch_size)
    for i in range(ftArray.size(0)):
      ft = ftArray[i, :]
      if ft[t] == 0: # 発火が規定回数以下なら
        if t == 0:#時間tが0のとき
          voltageGap = self.numFiringTime * torch.sum((self.threshold - ys[:, i, fireIndex])**2)
          costArrayex2[i] = costArrayex2[i] + self.gapWeight * voltageGap
          Iextx = 2 * alphamax - (alphamax - alphamin) * 0.5 * (torch.tanh(self.theta[t][:, 0]) + 1.0)
          costArrayex2[i] = costArrayex2[i] + torch.sum(torch.square(Iextx)) * 1 * self.gapWeight
        else:
          voltageGap = (self.numFiringTime - t) * torch.sum((self.c + (self.threshold - self.c) * (t + 1) - ys[math.ceil(ft[t - 1]/h):, i, fireIndex])**2)
          costArrayex2[i] = costArrayex2[i] + self.gapWeight * voltageGap
          Iextx = 2 * alphamax - (alphamax - alphamin) * 0.5 * (torch.tanh(self.theta[t][:, 0]) + 1.0)
          costArrayex2[i] = costArrayex2[i] + torch.sum(torch.square(Iextx)) * 1 * self.gapWeight
      else: # 発火していたら
        costArrayex1[i] = costArrayex1[i] + (ft[t]-self.idealFiringTime[t])**2
      if ft[t + 1] > 0: # 発火が規定回数より多ければ
        voltageGap = ft[t+1] * torch.sum((ys[math.ceil(ft[t]/h):, i, fireIndex] - self.c - (self.threshold - self.c) * (t+1))**2)
        costArrayex3[i] = costArrayex3[i] + self.gapWeight2 * voltageGap
      costArray[i] = costArrayex1[i] + costArrayex2[i] + costArrayex3[i]
    print(torch.sum(costArrayex1))
    print(torch.sum(costArrayex2))
    print(torch.sum(costArrayex3))
    return costArray

  def printCost(self, ys, x, tstr):#コストを別ファイルに保存するためのメソッド
    ftArray = self.firingTime(ys, self.numFiringTime-1)
    costArray = torch.zeros(batch_size)
    costArrayex1 = torch.zeros(batch_size)
    costArrayex2 = torch.zeros(batch_size)
    costArrayex3 = torch.zeros(batch_size)
    tfile = 'text_' + str(x) + '.txt'
    for i in range(ftArray.size(0)):
      ft = ftArray[i, :]
      for j in range(self.numFiringTime):
        if ft[j] == 0: # 発火が規定回数以下なら
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
        else: # 発火していたら
          costArrayex1[i] = costArrayex1[i] + (ft[j]-self.idealFiringTime[j])**2
      if ft[self.numFiringTime] > 0: # 発火が規定回数より多ければ
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
      if yv[i, 0] - self.threshold > 0:#最初の時間で筋電位が閾値以上のとき1回発火させ、筋電位をリセット
        numFire[i, 0] = 1
        yv[i, 0] = yv[i, 0] - (self.threshold - self.c)
      for j in range(yv.size(1) - 1):
        numFire[i, j+ 1] = numFire[i, j]
        yv[i, j+ 1] = yv[i, j+ 1] - numFire[i, j+ 1] * (self.threshold - self.c)
        if yv[i, j+ 1] - self.threshold > 0:
          numFire[i, j+ 1] = numFire[i, j+ 1] + 1
          yv[i, j+ 1] = yv[i, j+ 1] - (self.threshold - self.c)
    plt.figure()#0番目のyvのみをplot
    plt.plot(ts, yv[0,:])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    plt.savefig(fname)
    shutil.move(fname, 'dir_1_' + tstr)
    plt.show()
    # データを保存
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
    plt.figure()#全てのyvをプロット
    for i, sample in enumerate(yv):
      #sampleの+10より上から-10未満になる所を記録
      for k in range(len(sample)):
        if k != 0:
          if sample[k-1]>10 and sample[k]<-10:
            fire_time.append(ts[k-1])
      if i < 10:  # 最初の10個のサンプルのみプロット
        plt.plot(ts, sample, label=f'sample {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    #plt.legend()
    plt.savefig(fname)
    shutil.move(fname, 'dir_1_' + tstr)
    plt.show()

    fire_time_diff = []
    for j in range(len(fire_time)):
      fire_time_diff.append(-self.idealFiringTime[j%self.numFiringTime]+fire_time[j])
    print(fire_time_diff)
    np_time_diff = np.array(fire_time_diff)
    # 余りごとにデータを分割
    # インデックスに基づいてデータを分割
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
    # 各グループのデータをリストにまとめる
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

    # DataFrameに変換（NaNで埋める）
    df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

    # CSVファイルとして保存
    csv_filename = f"time_difference_groups_{fnum}.csv"
    df.to_csv(csv_filename, index=False)
    shutil.move(csv_filename, 'dir_1_' + tstr)
    # データを保存
    self.save_data_for_saveplot(ts, samples, tstr, fname)

  def save_data_for_saveplot(self, ts, samples, tstr, fname):
      # 時間ステップとサンプルデータをCPU上に移動
      ts = ts.cpu()
      yv = copy.deepcopy(samples[:, :, 0].squeeze().t().cpu())
      numFire = torch.zeros(batch_size, t_size)
      fire_time = []

      # サンプルデータの処理
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

      # 発火タイミングを記録
      for i, sample in enumerate(yv):
          for k in range(len(sample)):
              if k != 0 and sample[k - 1] > 10 and sample[k] < -10:
                  fire_time.append(ts[k - 1])

      # 保存するデータの構造を作成
      data_to_save = {
          'time_steps': ts.numpy(),
          'processed_samples': yv.numpy(),
          'numFire': numFire.numpy(),
          'fire_time': fire_time,
          'threshold': self.threshold,
          'c_value': self.c,
      }

      # pickle形式でデータを保存
      save_path = fname.replace('.pdf', '.pkl')
      with open(save_path, 'wb') as f:
          pickle.dump(data_to_save, f)

      shutil.move(save_path, f'dir_1_{tstr}/saveplot_{save_path}')

      print(f"データを {save_path} に保存しました")

  def save_data_for_uniplot(self, ts, samples, tstr, fname):
      # データをCPU上に移動（必要に応じて）
      ts = ts.cpu()
      yv = copy.deepcopy(samples[:, :, 0].squeeze().t().cpu())
      numFire = torch.zeros(batch_size, t_size)

      # サンプルデータの処理
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

      # 保存するデータの構造を作成
      data_to_save = {
          'time_steps': ts.numpy(),        # 時間軸データ
          'first_processed_sample': yv[0, :].numpy(),  # 最初のサンプルデータ（閾値処理後）
          'threshold': self.threshold,    # 閾値
          'c_value': self.c,              # 定数c
      }

      # データをpickle形式で保存
      save_path = fname.replace('.pdf', '.pkl')  # ファイル名を.pickle形式に変更
      with open(save_path, 'wb') as f:
          pickle.dump(data_to_save, f)
      shutil.move(save_path, f'dir_1_{tstr}/uniplot_{save_path}')
      print(f"uniデータを uniplot{save_path} に保存しました") 

def learnIz(a, b, c, d, fnum, tstr):#学習を行う関数
  # 初期値
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
  
  # 学習前を表示
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

  # コスト関数の値を表示
  print(loss)

  optimizer = optim.Adam(sde.parameters(), betas = (0.2, 0.36), lr=lr) # オプティマイザを指定
  epochNum = 500 #学習回数
  #showGraphNum = 1 # 学習中にグラフを描画する回数

  n = 0
  aLoss = torch.zeros(epochNum * sde.numFiringTime).detach().numpy()

  #torch.autograd.set_detect_anomaly(True)

  for l in range(sde.numFiringTime):
    if l > 0:
      optimizer.zero_grad() 
      sde.theta[l-1].requires_grad_(False)
      sde.theta[l].requires_grad_(True)
      sde.theta[l-1] = copy.deepcopy(sde.theta[l-1]) # thetaを更新されないようにする
      m = 0
    for k in tqdm(range(epochNum)):
      # 以下でグラフを描画
      print(n + 1)
      #if np.isin(k, np.linspace(0, epochNum, showGraphNum).astype(int)): 
        #with torch.no_grad():
          #ys = torchsde.sdeint(sde, y0, ts, method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
        #sde.plot(ts, ys, xlabel='$t$', ylabel='$Y_t$')
        
      # パラメータ更新前のおまじない
      optimizer.zero_grad() 

      # コスト関数を計算
      loss = sde.cost(torchsde.sdeint(sde, y0, ts, method='euler'), l).sum(0)
      # コスト関数の値を表示
      print(loss) 

      n = n + 1

      # コスト関数を小さくする方向の勾配を計算して，勾配に従ってパラメータ更新
      loss.backward()
      
      print("Iext:" + str(fnum))
      tmpTheta = copy.deepcopy(sde.theta[0])
      for i in range(sde.numFiringTime-1):
        tmpTheta2 = copy.deepcopy(sde.theta[i+1])
        tmpTheta = torch.cat((tmpTheta, tmpTheta2), 0)
      #tmpTheta = copy.deepcopy(sde.theta[l])
      Iext = alphamin + (alphamax - alphamin) * 0.5 * (torch.tanh(tmpTheta[0:t_size, :]) + 1.0).detach().numpy()
      print(Iext.squeeze())
      print("gradient:" + str(fnum))
      print(sde.theta[l].grad.squeeze())
      #inputの途中経過の保存、現在は1つの発火に対して1/4ずつ4回を保存
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
        # データ保存用のコード（pickleを使ってデータを保存する部分）
        data_to_save = {
            'ts': ts.numpy(),  # 時間ステップ
            'theta': [sde.theta[i].detach().numpy() for i in range(sde.numFiringTime)],  # sde.theta[i] の全てのデータ
            'fnum': fnum,  # fnum の値
            'l': l,  # l の値
            'epochNum': epochNum,  # epochNum の値
        }

        # 保存先ディレクトリ
        save_dir = f'dir_1_{tstr}'  # dir_1_ + tstr のディレクトリ

        # pickle形式でデータを保存
        save_path = f"{save_dir}/input_data_{fnum}_{l}_mid_input_data_{fnum}_{l}_epoch_{k}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"データを {save_path} に保存しました")

      # パラメータの更新前にパラメータを表示
      optimizer.step() 

  # 膜電位の変化
  y0 = torch.cat((
      torch.full(size=(batch_size, 1), fill_value=v0),
      torch.full(size=(batch_size, 1), fill_value=u0)), 1)
  with torch.no_grad():
      ys = torchsde.sdeint(sde, y0, ts, method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
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

  # 時間ステップ (ts) と入力データ (Iext) を保存
  data_to_save = {
      'ts': ts.numpy(),  # 時間ステップ ts
      'Iext': Iext,  # 入力データ Iext
      'fnum': fnum,  # fnum の値
  }

  # 保存先ディレクトリ (dir_1_{tstr})
  save_dir = f'dir_1_{tstr}'  # 保存するディレクトリ名

  # pickle形式でデータを保存
  save_path = f"{save_dir}/input_data_{fnum}.pkl"
  with open(save_path, 'wb') as f:
      pickle.dump(data_to_save, f)
  print(f"データを {save_path} に保存しました")

  # 発火時刻
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