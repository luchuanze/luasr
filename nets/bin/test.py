
import torch
import math
import collections
# pe = torch.zeros(x.size(1), self.dim, dtype=torch.float32)
# position = torch.arange(0, x.size(1), dtype=torch.float32).unsequeeze(1)
# div_term = torch.exp(
#     torch.arange(0, self.dim, 2, dtype=torch.float32)
#     * -(math.log(10000.0) / self.dim)
# )

testdict = {'b': 2, 'c': 5, 'a': 1}
ordict = collections.OrderedDict([('b', 2), ('c', 5), ('a', 1)])

print(ordict)




size1 = 5
dim = 4
pe = torch.zeros(5, dim, dtype=torch.float32)
position = torch.arange(0, size1, dtype=torch.float32).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, dim, 2, dtype=torch.float32)
    * -(math.log(10000.0) / dim)
)
print(position)
print(div_term)
print(position * div_term)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

print(pe.size(0))

print(pe)