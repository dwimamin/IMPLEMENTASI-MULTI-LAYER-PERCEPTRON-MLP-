# -*- coding: utf-8 -*-

import numpy as py
import pandas as pd
import math
import matplotlib.pyplot as plt

idx = ['x1','x2','x3','x4','types']
df = pd.read_csv('iris2.csv',names=idx)

data = df.head(150).values.tolist()

for i in data:
  if (i[4]=='Iris-setosa'):
    i.append(0)
    i.append(0)
  elif (i[4]=='Iris-versicolor'):
    i.append(0)
    i.append(1)
  else:
    i.append(1)
    i.append(0)

arrTrain = data[0:40] + data[50:90] + data[100:140]
arrVal = data[40:50] + data[90:100] + data[140:150]

train = arrTrain[:]
val = arrVal[:]

err_train= []
err_val = []
acc_train = []
acc_val = []

theta1 = [0.1,0.2,0.3,0.4]
dtheta1 = [0.0,0.0,0.0,0.0]
theta2 = [0.1,0.2,0.3,0.4]
dtheta2 = [0.0,0.0,0.0,0.0]
theta3 = [0.1,0.2]
dtheta3 = [0.0,0.0]
theta4 = [0.1,0.2]
dtheta4 = [0.3,0.3]
bias1 = 0.2
dbias1 = 0.0
bias2 = 0.3
dbias2 = 0.0
bias3 = 0.2
dbias3 = 0.0
bias4 = 0.3
dbias4 = 0.3

def Resulth1(x):
  res = bias1
  global theta1
  for i in range(4):
    res += (x[i]*theta1[i])
  return res

def Act1(x):
     
  return 1/(1 + math.exp(-x) )

def Resulth2(x):
  res2 = bias2
  for i in range(len(x)):
    global theta2
    res2 += (x[i]*theta2[i])
  return res2

def Act2(x):
  
  return 1/(1+math.exp(-x))

def Result1(x,y):
  global theta3
  global bias3
  total = bias3
  total += x*theta3[0] + y*theta3[1]
  return total

def ActOut1(x):
    return 1/(1+math.exp(-x))

def Result2(x,y):
  global theta4
  global bias4
  total = bias4
  total += x*theta4[0] + y*theta4[1] 
  return total

def ActOut2(x):
  return 1/(1+math.exp(-x))

def Predict(act):
  if(act>0.5):
    return 1
  else:
    return 0

def Error(trg,act):
  return math.pow((trg-act),2)

def Dtheta1Update(res1,trg1,act1, res2, trg2, act2, resh, acth):
  global dtheta1
  sum = 0.0
  sum += (act1-trg1)*act1*(1-act1)*res1
  sum += (act1-trg2)*act2*(1-act2)*res2
  dtheta = sum*acth*resh
  return dtheta1
                                  
def Dtheta2Update(res1,trg1,act1, res2, trg2, act2, resh, acth):
  global dtheta2
  sum = 0.0
  sum += (act1-trg1)*act1*(1-act1)*res1
  sum += (act1-trg2)*act2*(1-act2)*res2
  dtheta = sum*acth*resh
  return dtheta2

def Dtheta3Update(x,y,trg,act):
  global dtheta3
  dtheta3[0] =  x * (act-trg) * (1-act) * act
  dtheta3[1] =  y * (act-trg) * (1-act) * act

def Dtheta4Update(x,y,trg,act):
  global dtheta4
  dtheta4[0] =  x * (act-trg) * (1-act) * act
  dtheta4[1] =  y * (act-trg) * (1-act) * act
    
def Dbias1Update(trg,act):
  global dbias
  dbias = (act-trg) * (1-act) * act

def Dbias2Update(trg,act):
  global dbias
  dbias = (act-trg) * (1-act) * act

def Dbias3Update(trg,act):
  global dbias
  dbias = (act-trg) * (1-act) * act

def Dbias4Update(trg,act):
  global dbias
  dbias = (act-trg) * (1-act) * act

def Theta1Update(lr):
  global theta1
  for i in range(4):
    theta1[i] = theta1[i] - (lr*dtheta1[i])

def Theta2Update(lr):
  global theta1
  for i in range(4):
    theta2[i] = theta1[i] - (lr*dtheta2[i])

def Theta3Update(lr):
  global theta3
  for i in range(2):
    theta3[i] = theta4[i] - (lr*dtheta4[i])

def Theta4Update(lr):
  global theta4
  for i in range(2):
    theta4[i] = theta4[i] - (lr*dtheta4[i])

def Bias1Update(lr):
  global bias1
  bias1 -= (lr*dbias1)
def Bias2Update(lr):
  global bias2
  bias2 -= (lr*dbias2)
def Bias3Update(lr):
  global bias3
  bias3 -= (lr*dbias3)
def Bias4Update(lr):
  global bias4
  bias4 -= (lr*dbias4)

def main(lr):
  for i in range(100):   #n epoch
    sum_err_train, sum_err_val, sum_acc_train, sum_acc_val, Sumtotal, total, total2, tp_tn, tp_tn2 = 0,0,0,0,0,0,0,0,0
    #   train
    for k in range(120):
      
      resh1 = Resulth1(train[k][0:4])
      act1 = Act1(resh1)
      pred1 = Predict(act1)
      resh2 = Resulth2(train[k][0:4])
      act2 = Act2(resh2)
      pred2 = Predict(act2)
      res1 = Result1 (act1, act2)
      actout1 = ActOut1(res1)
      pred3 = Predict(actout1)
      res2 = Result2 (act1, act2)
      actout2 = ActOut2(res2)
      pred4 = Predict(actout2)
      
      Dtheta1Update(res1,train[k][5],actout1, res2, train[k][6],actout2,resh1, act1)
      Dbias1Update(train[k][5],actout1)
      Theta1Update(lr)
      Bias1Update(lr)
           
      Dtheta2Update(res1,train[k][5],actout1, res2, train[k][6],actout2,resh2, act2)
      Dbias2Update(train[k][6],actout2)
      Theta2Update(lr)
      Bias2Update(lr)
           
      Dtheta3Update(act1, act2, train[k][5],actout1)
      Dbias3Update(train[k][5],actout1)
      Theta3Update(lr)
      Bias3Update(lr)
      
      Dtheta4Update(act1, act2, train[k][6],actout2)
      Dbias4Update(train[k][6],actout2)
      Theta4Update(lr)
      Bias4Update(lr)
      
      
      total+=Error(train[k][5],actout1)
      total+=Error(train[k][6],actout2)
      
      if(pred3==train[k][5] and pred4==train[k][6]):
        tp_tn+=1     
      
    sum_err_train += total/120
    sum_acc_train += (tp_tn/120)
    
    
    # validation
    for k in range(30):
      
      act1 = Act1(Resulth1(val[k][0:4]))
      pred1 = Predict(act1)
      
      act2 = Act2(Resulth2(val[k][0:4]))
      pred2 = Predict(act2)
      
      actout1 = ActOut1(Result1 (act1, act2))
      pred3 = Predict(actout1)
      
      actout2 = ActOut2(Result2 (act1, act2))
      pred4 = Predict(actout2)
      
      total2 += Error(val[k][5],actout1)
      total2 += Error(val[k][6],actout2)
      
      if(pred3==val[k][5] and pred4==val[k][6]):
        tp_tn2+=1
      
      
    sum_err_val += total2/30
    sum_acc_val += (tp_tn2/30)

    err_train.append(sum_err_train)
    err_val.append(sum_err_val)
    acc_train.append(sum_acc_train)
    acc_val.append(sum_acc_val)
  
  plt.figure(1)
  plt.plot(acc_train,'r-', label='train')
  plt.plot(acc_val,'y-', label='validasi')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(loc='upper right')

  plt.figure(2)
  plt.plot(err_train,'r-', label='training')
  plt.plot(err_val,'y-', label='validasi')
  plt.xlabel('epoch')
  plt.ylabel('error')
  plt.legend(loc='upper right')
  plt.show()

#main(0.1)
main(0.8)

