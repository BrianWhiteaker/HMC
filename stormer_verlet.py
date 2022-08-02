#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:32:00 2022

@author: Dub
"""
import torch
from tensor_type import Tensor
from typing import List, Tuple, Callable, Union, Optional


class SV:
    '''
        Assumes: 
        -potential function is given
    '''
    def __init__(self,
                 U: Callable,
                 h: float,
                 dim: int):
        
        self.U   = U
        self.h   = h
        self.dim = dim
            
    def gradient(self,
                 func: Callable,
                 x:    Tensor) -> Tuple[Tensor, Tensor]:
        '''
        dH calculates callable function and its gradient.
        x: tensor
        Mat: tensor (mass matrix)
        func: is a function to calculate
        returns function output and gradient
        '''
        
        x.requires_grad_(True)    # track math operations
        x.retain_grad()           # keep gradient after backward()
        out    = func(x)          # calculate given function
        out.backward(gradient=torch.ones(out.size()))  # Calculate grads
        x_grad = x.grad.data      # get gradients only
        x.grad = None             # reset gradient 
        x.requires_grad_(False)   # stop tracking
        return out.detach().clone(), x_grad 

    def sv1(self,
            p: Tensor,
            q: Tensor):
                                  # half-step p
        _, g1 = self.gradient(self.U, q)
        p     -= .5*self.h*g1
                                  # full-step q
        q     += self.h*p
                                  # half-step p
        _, g2 = self.gradient(self.U, q)
        p     -= .5*self.h*g2
        
        return -p,q

