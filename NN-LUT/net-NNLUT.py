#! /usr/bin/python3
# vim: set sw=2 sts=2 ts=8 et:

import torch
import torch.nn as nn
import random
import numpy as np
import math
import matplotlib.pyplot as plt

import sys
import os
from optparse import OptionParser

PROG = os.path.basename(sys.argv[0])


def calc_GELU(x):
#  val = torch.tensor(x)
#  f = nn.GELU()
#  return f(val).tolist()
  return  (0.5*x*(1+np.tanh(math.sqrt(2/3.1416)*(x+0.444715*pow(x,3)))))

def calc_swish(x):
  #val = torch.tensor(x)
  #f = nn.Sigmoid()
  #return f(val).tolist()*x
  return x/(1+math.exp(-1*x)) 

def calc_softplus(x):
  #val = torch.tensor(x)
  #f = nn.Softplus()
  #return f(val).tolist()
  return math.log(1+math.exp(x)) 

if __name__ == '__main__':
  help='''usage: %prog [options]", version=1.0'''

  parser = OptionParser(usage=help)
  parser.add_option("-o", "--output", type="string", help="output CSV file <OUTPUT> of the error")
  parser.add_option("-f", "--function", type="string", help="specify the function to approximate (GELU, swish, softplus)")
  parser.add_option("-r", "--rateLearning", type="float", default=0.01, help="set the learning rate (default: 0.01)")
  parser.add_option("-b", "--batchSize", type="int", default=50000, help="specifiy the size of the trainig dataset (default: 30000)")
  parser.add_option("-m", "--maxAbsError", type="float", default=0.00000095367, help="maximum absolute error for approximation (default: 0.00000095367)")
  parser.add_option("-e", "--epochs", type="int", default=2000, help="epochs for training the NN (default: 2000)")
  parser.add_option("-n", "--neuronsHidden", type="int", default=1000, help="number of hidden neurons in the hidden layer (default: 1000)")
  parser.add_option("-p", "--plots", type="string", default="off", help="set on or off the plots (default: off)")

  (options, args) = parser.parse_args()

  if not options.output:
    print("%s: missing --output option!" % PROG, file=sys.stderr)
    exit(-1)

  if not options.function:
    print("%s: missing --function option!" % PROG, file=sys.stderr)
    exit(-1)

  # exact function values
  x_vals = np.linspace(-5,5,3000)
  y_GELU = calc_GELU(x_vals)
  y_swish = []
  y_softplus = [] 
  for i in range(len(x_vals)):
    y_swish.append(calc_swish(x_vals[i]))
    y_softplus.append(calc_softplus(x_vals[i]))

  y_vals = []
  if options.function == "GELU":
    y_vals = y_GELU
  elif options.function == "softplus":
    y_vals = y_softplus   
  else:
    y_vals = y_swish

  # defining testing data
  testing_vals = torch.FloatTensor(x_vals)
  testing_vals = torch.reshape(testing_vals, (len(testing_vals),1))

  # setting the seed
  random.seed(10)

  # one input
  n_input = 1
  # start with 10 nodes in the hden layer
  n_hidden = 10
  # number of nodes at the output layer
  n_out = 1
  # batch_size (size of the training set)
  batch_size = options.batchSize
  # learning rate hyperparamter
  learning_rate = options.rateLearning

  #generate training dataset
  training = []
  labels   = []

  for i in range(batch_size):
    val = random.uniform(-5, 5)
    training.append(val)

    if options.function == "GELU":
      labels.append(calc_GELU(val))
    elif options.function == "softplus":
      labels.append(calc_softplus(val))
    else:
      labels.append(calc_swish(val))

  training = torch.FloatTensor(training)
  training = torch.reshape(training, (len(training),1))

  labels = torch.FloatTensor(labels)
  labels = torch.reshape(labels, (len(labels),1))

  max_error = 10000000
  max_abs_error = options.maxAbsError
  losses= []
  prediction = []

  n_hidden_set = [2000,5000]
  
  n_hidden = options.neuronsHidden
  oFile = open(options.output, "w")
  model = nn.Sequential(nn.Linear(n_input, n_hidden),
                       nn.ReLU(),
                       nn.Linear(n_hidden, n_out))
  
  loss_function = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  
  losses = []

  for epoch in range(options.epochs):
    pred_y = model(training)
    loss = loss_function(pred_y, labels)
    losses.append(loss.item())
    
    model.zero_grad()
    loss.backward()
    
    optimizer.step()

  # predicting on testing dataset
  prediction = model(testing_vals)
  print(prediction.shape)
  prediction = prediction.tolist()

  error = []
  for i in range(len(prediction)):
    error.append(abs(prediction[i][0]-y_vals[i]))
    oFile.write(str(abs(prediction[i][0]-y_vals[i]))+"\n")
  
  max_error = max(error)
  oFile.write("Testing hiden nodes: "+str(n_hidden)+" with max error: "+str(max_error)+" learning rate "+str(options.rateLearning)+" epochs "+str(options.epochs)+"\n")  
  oFile.close()
  if math.isnan(max_error):
    print(prediction)
    print(error)
  print("Testing hiden nodes: "+str(n_hidden)+" with max error: "+str(max_error)+" and max absolute error:"+str(max_abs_error))


  if options.plots == "on":
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
   
    plt.figure(200)
    plt.plot(x_vals,y_vals,label="exact %s"%(options.function))
    plt.plot(x_vals,prediction,label="approximate %s"%(options.function))
    plt.ylabel("f(x)")
    plt.xlabel("x")
    plt.legend(loc='best')
    plt.title("Approximating %s"%(options.function))
    
    plt.figure(300)
    plt.plot(x_vals,error,label="approximation error")
    plt.ylabel("absolute error")
    plt.xlabel("x")
    plt.title("Absolute error margins")
    
    plt.show()

