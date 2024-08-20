#!/usr/bin/env python3

import torch

"""
qcd_ml.qcd.static
=================

Exports ``gamma`` which provide the euclidean gamma matrices. 
The matrices are chosen to be identical to the default choice 
in `lehner/gpt <https://github.com/lehner/gpt>`_.
"""

gamma = [torch.tensor([[0,0,0,1j]
                      ,[0,0,1j,0]
                      ,[0,-1j,0,0]
                      ,[-1j,0,0,0]], dtype=torch.cdouble)
         , torch.tensor([[0,0,0,-1]
                        ,[0,0,1,0]
                        ,[0,1,0,0]
                        ,[-1,0,0,0]], dtype=torch.cdouble)
         , torch.tensor([[0,0,1j,0]
                        ,[0,0,0,-1j]
                        ,[-1j,0,0,0]
                        ,[0,1j,0,0]], dtype=torch.cdouble)
         , torch.tensor([[0,0,1,0]
                        ,[0,0,0,1]
                        ,[1,0,0,0]
                        ,[0,1,0,0]], dtype=torch.cdouble)
         ]
