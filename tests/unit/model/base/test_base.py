import pytest 
import numpy as np
from typing import Tuple, Callable, Any, Dict
from numpy.typing import ArrayLike

from gnomix.model.base.base import Base

from gnomix.model.base.models import (
    LogisticRegressionBase,
    XGBBase,
    StringKernelBase,
)

BASE_MODELS = [LogisticRegressionBase, XGBBase, StringKernelBase]

def get_binary_toy_data(C: int, W: int) -> Tuple[Tuple[ArrayLike, ArrayLike], Dict[str, Any]]:

    N_0 = 10
    X_0 = np.zeros((N_0, C))
    y_0 = np.zeros((N_0, W), dtype=int)

    N_1 = 10
    X_1 = np.ones((N_1, C))
    y_1 = np.ones((N_1, W), dtype=int)

    X = np.concatenate([X_0, X_1])
    y = np.concatenate([y_0, y_1])

    meta_data = {"num_ancestry": 2}

    return (X, y), meta_data

def get_multilabel_toy_data(C: int, W: int) -> Tuple[Tuple[ArrayLike, ArrayLike], Dict[str, Any]]:

    N_0 = 10
    X_0 = np.zeros((N_0, C))
    y_0 = np.zeros((N_0, W), dtype=int)

    N_1 = 10
    X_1 = np.ones((N_1, C))
    y_1 = np.ones((N_1, W), dtype=int)

    N_2 = 10
    X_2 = np.ones((N_2, C))
    y_2 = np.ones((N_2, W), dtype=int)

    X = np.concatenate([X_0, X_1, X_2])
    y = np.concatenate([y_0, y_1, y_2])

    meta_data = {"num_ancestry": 3}
    
    return (X, y), meta_data

@pytest.mark.parametrize("base_model_class", BASE_MODELS)
@pytest.mark.parametrize("get_toy_data", [get_binary_toy_data, get_multilabel_toy_data])
@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("base_multithread", [False, True])
@pytest.mark.parametrize("context", [0, 50])
@pytest.mark.parametrize("C", [299, 300, 301])
def test_binary_base_train_inference(
    base_model_class: Base,
    get_toy_data: Callable,
    vectorize: bool,
    base_multithread: bool,
    context: int,
    C: int
):

    M = 100
    W = int(np.ceil(C / M))
    (X, y), meta_data = get_toy_data(C, W)

    base = base_model_class(
        C=C,
        M=M,
        W=W,
        A=meta_data["num_ancestry"],
        vectorize=vectorize,
        base_multithread=base_multithread,
        context=context
    )

    base.train(X, y)
    y_pred = base.predict(X)
    assert np.all(y==y_pred)


@pytest.mark.parametrize("array, pad_width, expected_output", [
    (
        np.array([
            [1,2,3],
            [4,5,6]
        ]),
        0,
        np.array([
            [1,2,3],
            [4,5,6]
        ]),
    ),
    (
        np.array([
            [1,2,3],
            [4,5,6]
        ]),
        1,
        np.array([
            [1,1,2,3,3],
            [4,4,5,6,6]
        ]),
    ),
    (
        np.array([
            [1,2,3],
            [4,5,6]
        ]),
        2,
        np.array([
            [2,1,1,2,3,3,2],
            [5,4,4,5,6,6,5]
        ]),
    )
])
def test_pad_with_reflection_along_first_axis(
    array: ArrayLike,
    pad_width: int,
    expected_output: ArrayLike
) -> None:
    output = Base.pad_with_reflection_along_first_axis(
        array=array, 
        pad_width=pad_width
    )
    assert np.all(output == expected_output)
