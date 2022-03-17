from torch import nn, tanh, relu

from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms


def build_maf(dim=1, num_transforms=8, context_features=None, hidden_features=128):

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=dim,
                        hidden_features=hidden_features,
                        context_features=context_features,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=tanh,
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=dim),
                ]
            )
            for _ in range(num_transforms)
        ]
    )

    distribution = distributions_.StandardNormal((dim,))
    neural_net = flows.Flow(transform, distribution)
    
    return neural_net