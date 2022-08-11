import torch

from nets.core.predictor_stateless import Predictor


def test_predictor_stateless():
    vocab_size = 3
    blank_id = 0
    embedding_dim = 128
    context_size = 4

    predictor = Predictor(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        blank_id=blank_id,
        context_size=context_size
    )

    N = 100
    U = 20
    x = torch.randint(low=0, high=vocab_size, size=(N, U))
    y = predictor(x)
    assert y.shape == (N, U, embedding_dim)

    x = torch.randint(low=0, high=vocab_size, size=(N, context_size))
    y = predictor(x, need_pad=False)
    assert y.shape == (N, 1, embedding_dim)


def main():
    test_predictor_stateless()


if __name__ == "__main__":
    main()
