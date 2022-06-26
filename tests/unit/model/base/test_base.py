import pytest 
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike

from gnomix.model.base.base import Base

from gnomix.model.base.models import (
    LogisticRegressionBase,
    XGBBase,
    StringKernelBase
)

BASE_MODELS = (LogisticRegressionBase, XGBBase, StringKernelBase)

def get_binary_toy_data(chm_len: int, num_windows: int) -> Tuple[ArrayLike, ArrayLike]:

    N_0 = 10
    X_0 = np.zeros((N_0, chm_len))
    y_0 = np.zeros((N_0, num_windows), dtype=int)

    N_1 = 10
    X_1 = np.ones((N_1, chm_len))
    y_1 = np.ones((N_1, num_windows), dtype=int)

    X = np.concatenate([X_0, X_1])
    y = np.concatenate([y_0, y_1])

    return X, y

@pytest.mark.parametrize("base_model_class", BASE_MODELS)
@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("base_multithread", [False, True])
@pytest.mark.parametrize("context", [0, 50])
@pytest.mark.parametrize("chm_len", [299, 300, 301])
def test_binary_base_train_inference(
    base_model_class: Base,
    vectorize: bool,
    base_multithread: bool,
    context: int,
    chm_len: int
):
    window_size = 100
    num_windows = int(np.ceil(chm_len / window_size))
    X, y = get_binary_toy_data(chm_len, num_windows)

    base = base_model_class(
        chm_len=chm_len,
        window_size=window_size,
        num_windows=num_windows,
        num_ancestry=2,
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
