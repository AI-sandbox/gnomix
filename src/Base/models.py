import numpy as np

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm

from src.Base.base import Base
from src.Base.string_kernel import string_kernel, random_string_kernel
from src.Base.string_kernel import string_kernel_singlethread, random_string_kernel_singlethread

class LogisticRegressionBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_multithread = True

        self.init_base_models(
            lambda : LogisticRegression(penalty="l2", C = 3., solver="liblinear", max_iter=1000)
        )


class XGBBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : XGBClassifier(
                n_estimators=20, max_depth=4, learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                thread=self.n_jobs, missing=self.missing_encoding, random_state=self.seed)
        )

class LGBMBase(Base):

    def __init__(self,  *args, **kwargs):
        
        from lightgbm import LGBMClassifier # This is to avoid requiring installation of lightgbm

        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : LGBMClassifier(
                n_estimators=20, max_depth=4, learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                n_jobs=self.n_jobs, random_state=self.seed) # use np.nan for missing encoding
        )

class RFBase(Base):

    def __init__(self,  *args, **kwargs):

        from sklearn.ensemble import RandomForestClassifier

        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : RandomForestClassifier(n_estimators=20,max_depth=4,n_jobs=self.n_jobs) 
        )

class KNNBase(Base):

    from sklearn.neighbors import KNeighborsClassifier

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : KNeighborsClassifier(n_neighbors=1)
        )

class SVMBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : svm.SVC(C=100., gamma=0.001, probability=True)
        )

class StringKernelBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert int(np.__version__.split(".")[1]) >= 20, "String kernel implementation requires numpy versions 1.20+"

        self.train_admix = False # save computation

        self.kernel = string_kernel if self.n_jobs!=1 else string_kernel_singlethread

        self.init_base_models(
            lambda : svm.SVC(kernel=self.kernel, probability=True)
        )


class RandomStringKernelBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert int(np.__version__.split(".")[1]) >= 20, "String kernel implementation requires numpy versions 1.20+"
        
        self.train_admix = False # save computation

        self.kernel = random_string_kernel if self.n_jobs!=1 else random_string_kernel_singlethread

        self.init_base_models(
            lambda : svm.SVC(kernel=self.kernel, probability=True)
        )
