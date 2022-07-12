import pytest 
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from numpy.typing import ArrayLike

from gnomix.model.smooth.xgb import XGBSmoother
from gnomix.model.smooth.crf import CRFSmoother

def get_binary_toy_data(W: int) -> Tuple[Tuple[ArrayLike, ArrayLike], Dict[str, Any]]:

    N_0 = 10
    X_0 = np.zeros((N_0, W, 2))
    X_0[:, :, 0] = 1
    y_0 = np.zeros((N_0, W), dtype=float)

    N_1 = 10
    X_1 = np.zeros((N_1, W, 2))
    X_1[:, :, 1] = 1
    y_1 = np.ones((N_1, W), dtype=int)

    X = np.concatenate([X_0, X_1])
    y = np.concatenate([y_0, y_1])

    meta_data = {"num_ancestry": 2}

    return (X, y), meta_data

def get_multilabel_toy_data(W: int) -> Tuple[Tuple[ArrayLike, ArrayLike], Dict[str, Any]]:

    N = 10

    X_0 = np.zeros((N, W, 3))
    X_0[:, :, 0] = 1
    y_0 = np.zeros((N, W), dtype=float)

    X_1 = np.zeros((N, W, 3))
    X_1[:, :, 1] = 1
    y_1 = np.zeros((N, W), dtype=float) + 1

    X_2 = np.zeros((N, W, 3))
    X_2[:, :, 2] = 1
    y_2 = np.zeros((N, W), dtype=float) + 2

    X = np.concatenate([X_0, X_1, X_2])
    y = np.concatenate([y_0, y_1, y_2])

    meta_data = {"num_ancestry": 3}

    return (X, y), meta_data


@pytest.mark.parametrize("smoother_width", [75, 100, 125])
@pytest.mark.parametrize("get_toy_data", [get_binary_toy_data, get_multilabel_toy_data])
def test_xgb_smooth_train_inference(
    get_toy_data: Callable,
    smoother_width: int,
    W: int = 100
):

    (X, y), meta_data = get_toy_data(W)
    smoother = XGBSmoother(
        num_ancestry=meta_data["num_ancestry"],
        smoother_width=smoother_width
    )
    smoother.train(X, y)
    y_proba = smoother.predict_proba(X)
    y_pred = np.argmax(y_proba, axis=-1)

    assert np.all(y==y_pred)


@pytest.mark.parametrize("get_toy_data", [get_binary_toy_data, get_multilabel_toy_data])
def test_crf_smooth_train_inference(
    get_toy_data: Callable,
    W: int = 100
):

    (X, y), meta_data = get_toy_data(W)
    smoother = CRFSmoother(
        num_ancestry=meta_data["num_ancestry"]
    )
    smoother.train(X, y)
    y_proba = smoother.predict_proba(X)
    y_pred = np.argmax(y_proba, axis=-1)

    assert np.all(y==y_pred)
