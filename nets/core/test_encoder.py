
import torch
import numpy

from nets.core.encoder_transformer import (
     Transformer,
)

def test_transformer():
    num_features = 40
    num_classes = 87

    model = Transformer(input_size=num_features, output_dim=num_classes)

    batch_size = 31
    time1 = 8
    #x = torch.rand(batch_size, time1, num_features)
    # s = x.numpy()
    # print(s)
    # numpy.save('x_tensor.npy', s)

    s = numpy.load('x_tensor.npy', allow_pickle=True)
    x = torch.tensor(s)
    x_lens = torch.ones(batch_size, dtype=torch.int32) * time1

    y, _ = model(x, x_lens)

    print(y)