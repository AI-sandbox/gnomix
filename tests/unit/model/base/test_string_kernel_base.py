import pytest 
import numpy as np

from gnomix.model.base.base import Base
from gnomix.model.base.models import (
    StringKernelBase
)

def test_string_kernel_base():

    chm_len = 301
    window_size = 100
    num_windows = int(np.ceil(chm_len / window_size))
    num_ancestry = 2

    N_0 = 10
    X_0 = np.zeros((N_0, chm_len))
    y_0 = np.zeros((N_0, num_windows), dtype=int)

    N_1 = 10
    X_1 = np.ones((N_1, chm_len))
    y_1 = np.ones((N_1, num_windows), dtype=int)

    X = np.concatenate([X_0, X_1])
    y = np.concatenate([y_0, y_1])

    base = StringKernelBase(
        chm_len=chm_len, 
        window_size=window_size, 
        num_ancestry=num_ancestry
    )

    base.train(X, y)
    y_pred = base.predict(X)
    assert np.all(y==y_pred)

    
    