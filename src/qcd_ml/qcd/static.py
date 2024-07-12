#!/usr/bin/env python3

import torch

gamma = [torch.complex(
             torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.double),
             torch.tensor([[0,0,0,1]
                           ,[0,0,1,0]
                           ,[0,-1,0,0]
                           ,[-1,0,0,0]], dtype=torch.double)
             )
         , torch.complex(
             torch.tensor([[0,0,0,-1]
                           ,[0,0,1,0]
                           ,[0,1,0,0]
                           ,[-1,0,0,0]], dtype=torch.double),
             torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.double)
             )
         , torch.complex(
             torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.double),
             torch.tensor([[0,0,1,0]
                           ,[0,0,0,-1]
                           ,[-1,0,0,0]
                           ,[0,1,0,0]], dtype=torch.double)
             )
         , torch.complex(
             torch.tensor([[0,0,1,0]
                           ,[0,0,0,1]
                           ,[1,0,0,0]
                           ,[0,1,0,0]], dtype=torch.double),
             torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.double)
             )
         ]
