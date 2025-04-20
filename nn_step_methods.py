import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F




def Eulerstep(net, input_batch, time_step):
 output_1 = net(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1) 
  
def Directstep(net, input_batch, time_step):
  output_1 = net(input_batch.cuda())
  return output_1


def RK4step(net, input_batch, time_step):
 output_1 = net(input_batch.cuda())
 output_2 = net(input_batch.cuda()+0.5*output_1)
 output_3 = net(input_batch.cuda()+0.5*output_2)
 output_4 = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6


def PECstep(net, input_batch, time_step):
 output_1 = net(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_1))

def PEC4step(net, input_batch, time_step):
 output_1 = time_step*net(input_batch.cuda()) + input_batch.cuda()
 output_2 = input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_1))
 output_3 = input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_2))
 return input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_3))


class PEC_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(PEC_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
      output_1 = self.network(input_batch.to(self.device)) + input_batch.to(self.device)
      return input_batch.to(self.device) + self.time_step*0.5*(self.network(input_batch.to(self.device))+self.network(output_1))
    

class Euler_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Euler_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
      return input_batch.to(self.device) + self.time_step*(self.network(input_batch.to(self.device)))


class PEC4_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(PEC4_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
        temp_out_1 = self.network(input_batch.cuda())
        output_1 = self.time_step*temp_out_1 + input_batch.cuda()
        output_2 = input_batch.cuda() + self.time_step*0.5*(temp_out_1+self.network(output_1))
        output_3 = input_batch.cuda() + self.time_step*0.5*(temp_out_1+self.network(output_2))
        return input_batch.cuda() + self.time_step*0.5*(temp_out_1+self.network(output_3))


class RK4_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super().__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step


    def forward(self, input_batch):
        input_batch = input_batch.to(self.device)
        output_1 = self.network(input_batch)
        output_2 = self.network(input_batch+0.5*output_1)
        output_3 = self.network(input_batch+0.5*output_2)
        output_4 = self.network(input_batch+output_3)

        return input_batch + self.time_step*(output_1+2*output_2+2*output_3+output_4)/6

class Direct_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Direct_step, self).__init__()
        self.network = network
        self.device = device

    def forward(self, input_batch):
      return self.network(input_batch.to(self.device))

class Implicit_Euler_step(nn.Module):
    def __init__(self, network, device, num_iters, time_step = 1): 
        super(Implicit_Euler_step, self).__init__()
        self.network = network
        self.device = device
        self.num_iters = num_iters
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def implicit_euler(self, input_batch, output):
        return input_batch.to(self.device) + self.time_step*(self.network(output.to(self.device)))
    
    def change_num_iters(self, num_iters):
        self.num_iters = num_iters
        print('New iters: ',self.num_iters)
        return
    
    def forward(self, input_batch):
        output = input_batch.to(self.device) + self.time_step*(self.network(input_batch.to(self.device)))
        iter = 0
        while iter < self.num_iters:
           output = self.implicit_euler(input_batch, output)
           iter += 1
        return output