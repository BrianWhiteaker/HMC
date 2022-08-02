#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:00:04 2022

@author: Dub
"""

import torch
from tensor_type import Tensor
from typing import Tuple, Callable, Optional

class HMC():
    def __init__(self,
                 method:    Callable,
                 potential: Callable,
                 D:         int,
                 M:         Optional[Tensor]=None,
                 updateM:   bool=False):
        
        self.method = method  # Callable integration method
        self.U      = potential
        if(not None):
            self.M  = M
        else:
            self.M  = torch.eye(D)
        self.Mi     = torch.inverse(M)
        
 #%%       
    def K(self, p:  Tensor) -> Tensor:
        
        return torch.squeeze( (p.t()@self.Mi@p)*.5 )
    
 #%%   
    def dK(self, p:  Tensor) -> Tensor:
        
        return torch.squeeze( self.Mi@p)
    
 #%%       
    def H(self,
          p:  Tensor,
          q:  Tensor) -> Tensor:
        
        return torch.squeeze( self.K(p) + self.U(q))
    
 #%%   
    def alpha(self,
              p0:     Tensor,
              q0:     Tensor,
              pstar:  Tensor,
              qstar:  Tensor) -> float:
        
        a = torch.Tensor([1., torch.exp(self.H(p0,q0)-\
                                        self.H(pstar,qstar))])
        return (torch.min(a)).item()
    
 #%%   
    def accept(self,
               a:      float,
               q0:     Tensor,
               qstar:  Tensor):
        
        acceptFlag = False
        u = torch.rand(0,1)   # ?????? change issue
        if u < a:
            q0 = qstar
            acceptFlag = True
        else:
            q0 = q0
            acceptFlag = False
        return q0, acceptFlag
    
 #%%   
    def sample(self,
               q0: Tensor):
        
        p0 = torch.randn((2,1))        # sample normal dist
        Pp,Qp = self.method(p0,q0)     # propagate
        acc = self.alpha(p0,q0,Pp,Qp)  # get acceptance prob
        return  self.accept(acc, q0, Qp)    # accept/reject