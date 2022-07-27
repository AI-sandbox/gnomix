import numpy as np
from numpy.typing import ArrayLike
from xgboost import XGBClassifier
from typing import Dict, Tuple, Any, Optional

from gnomix.model.smooth.smoother import Smoother

class XGBSmoother(Smoother):

    def __init__(
        self,
        num_ancestries: int,
        num_features: Optional[int] = 75,
        xgb_config: Optional[Dict[str, Any]] = {},
        input_dtype: Optional[str] = "float32",
        proba_dtype: Optional[str] = "float32"
    ):
        self.num_features = num_features
        xgb_config.update({
            "num_class": num_ancestries,
            "objective": 'multi:softprob',
            "eval_metric": "mlogloss"
        })
        self.model = XGBClassifier(**xgb_config)
        self.input_dtype = input_dtype
        self.proba_dtype = proba_dtype

    def fit(self, B: ArrayLike, y: ArrayLike) -> None:
        B_slide, y_slide = self._slide_window(B, y)
        self.model.fit(B_slide, y_slide)

    def predict_proba(self, B: ArrayLike) -> ArrayLike:

        N, W, A = B.shape

        B_slide, _ = self._slide_window(B)
        proba_flat = self.model.predict_proba(B_slide)
        proba = proba_flat.reshape(N, W, A).astype(self.proba_dtype)
        
        return proba

    def predict(self, B: ArrayLike) -> ArrayLike:
        y_proba = self.predict_proba(B)
        y_pred = np.argmax(y_proba, axis=-1)
        return y_pred

    def _slide_window(self, B: ArrayLike, y: Optional[ArrayLike] = None) -> Tuple[ArrayLike, ArrayLike]:
        
        N, W, A = B.shape

        # pad it.
        pad = (self.num_features + 1) // 2
        pad_left = np.flip(B[:, 0:pad, :], axis=1)
        pad_right = np.flip(B[:, -pad:, :], axis=1)
        B_padded = np.concatenate([pad_left, B, pad_right], axis=1)

        # window it.
        B_slide = np.zeros((N, W, A * self.num_features), dtype=self.input_dtype)
        for ppl, dat in enumerate(B_padded):
            for w in range(W):
                B_slide[ppl, w, :] = dat[w:(w + self.num_features)].ravel()

        # reshape
        B_slide = B_slide.reshape(N * W, A * self.num_features)
        y_slide = None if y is None else y.reshape(N * W)

        return B_slide, y_slide
        