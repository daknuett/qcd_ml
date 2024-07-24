#!/usr/bin/env python3

import torch

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
