from typing import Union

import jax.numpy as jnp
from numpy import ndarray
from trax import layers as tl
from trax.layers import combinators as cb
from trax.layers.assert_shape import assert_shape
from trax.layers.combinators import Serial as Traxmodel
from trax.shapes import signature

Array = Union[jnp.ndarray, ndarray]


@assert_shape("bs->bd")
def makecustommodel(units: int):
    model = cb.Serial(tl.Dense(units), tl.Elu())
    return model


def summary(
    model: Traxmodel, X: Array, init: int = 1, counter: int = 0  # noqa N803
) -> Array:
    output = X  # noqa N803
    input = signature(output)
    if init == 1:
        print(
            f'{"layer":<23} {"input":<19} {"dtype":^7}    {"output":<19} {"dtype":^7}'  # noqa N803
        )
    for sub in model.sublayers:
        name = str(sub.name)
        if name == "":
            continue
        elif name == "Serial":
            output = summary(sub, output, init + 1, counter)
        else:
            output = sub.output_signature(input)
            print(
                f"({counter}) {str(sub.name):<19} {str(input.shape):<19}({str(input.dtype):^7}) | {str(output.shape):<19}({str(output.dtype):^7})"  # noqa E501
            )
        input = output
        counter += 1
    return output
