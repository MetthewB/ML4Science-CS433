import torch
import torch.nn.functional as F 
import numpy as np 


class TV_ISO():
    def __init__(self, device):

        self.filters1 = torch.Tensor([[[[1., -1.]]]]).double().to(device)
        self.filters2 = torch.Tensor([[[[1.], [-1.]]]]).double().to(device)

    def L(self, x, h): 
        conv_1 = F.conv2d(F.pad(x, (1, 1, 1, 0)), self.filters1)
        conv_2 = F.conv2d(F.pad(x, (1, 0, 1, 1)), self.filters2)

        L1_1 = F.pad(conv_1, (0, 1, 0, 1))
        L2_1 = F.pad(conv_2, (0, 1, 0, 1))

        Lx = torch.cat((L1_1, L2_1), dim=1) 

        return Lx * h 
    
    def Lt(self, y, h):


        Lt1y_1 = F.conv_transpose2d(y[:, 0:1, :-1, :-1], self.filters1)[:, :, 1:, 1:-1]
        Lt2y_1 = F.conv_transpose2d(y[:, 1:2, :-1, :-1], self.filters2)[:, :, 1:-1, 1:]

        
        Lty = Lt1y_1 + Lt2y_1 
        
        return Lty * h 
    

class prox_tv_iso(): 

    def __init__(self, device, box_const=None):
         
        self.device = device
        self.tv = TV_ISO(device)
        self.box_const = box_const
    
    def P_c(self, x):
        if self.box_const is None:
            return x
        else:
            return torch.clip(x, self.box_const[0], self.box_const[1])

    
    def eval(self, y, niter, lmbda, h=1, verbose=False, stop=-1):

        v_k = torch.zeros((1, 2, y.size(2)+2, y.size(3)+2), requires_grad=False, device=self.device).double()
        u_k = torch.zeros((1, 2, y.size(2)+2, y.size(3)+2), requires_grad=False, device=self.device).double()

        t_k = 1 

        alpha = 1 / (8 * lmbda)

        self.loss_list = list()
        loss_old = ((y)**2).sum() / 2
        
        for iters in range(niter):
            LTv = self.tv.Lt(v_k, h)
            Lpc = self.tv.L(self.P_c(y - lmbda * LTv), h)

            if verbose: 
                loss = ((y - self.P_c(y - lmbda * LTv))**2).sum() / 2 + lmbda * torch.norm(Lpc, dim=1, p=2).sum()
                self.loss_list.append(loss.item())
                
            u_kp1 = F.normalize(v_k + alpha * Lpc, eps=1, dim=1, p=2)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         


            u_k = u_kp1
            v_k = v_kp1
            t_k = t_kp1

            if stop > 0 and verbose:
                loss_new = loss
                crit = torch.abs(loss_new - loss_old)
                if (crit < (stop * loss_old)) and (iters > 3):
                    self.niters = iters + 1
                    break
                
                loss_old = loss_new

        c = y - lmbda * self.tv.Lt(u_k, h)
        c = self.P_c(c)

        return c