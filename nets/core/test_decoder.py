import torch

from nets.core.loss_transformer import add_sos, add_eos

x = torch.tensor([[1, 2, 3, 4, 5],
                 [4, 5, 6, 0, 0],
                 [7, 9, 9, 8, 0]], dtype=torch.int32)

y = add_eos(x, 10, 0, -1)
print(y)