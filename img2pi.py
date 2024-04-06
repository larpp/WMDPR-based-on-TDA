import numpy as np

from ripser import Rips
from persim import PersistenceImager
import collections
collections.Iterable = collections.abc.Iterable


# Change images to Persistence Images
def PI(defect, lpi=True):
    rips = Rips()
    dgms = rips.fit_transform(defect)
    H0_dgm = dgms[0] # H_0 : connected component
    H1_dgm = dgms[1] # H_1 : hole

    H0_dgm = H0_dgm[:-1,:] # inf 값 제외

    pimgr0 = PersistenceImager(pixel_size=1, birth_range=(0,10), pers_range=(0.0, 10))
    H0 = pimgr0.transform(H0_dgm).flatten() # 100차원 벡터

    pimgr1 = PersistenceImager(pixel_size=1, birth_range=(0,10), pers_range=(0.0, 10))
    H1 = pimgr1.transform(H1_dgm).flatten() # 100차원 벡터

    if lpi:
        pimgr = PersistenceImager(pixel_size=2, birth_range=(0,20), pers_range=(0.0, 20))
        pimgr.weight_params = {'n' : 0} # persistence에 대한 가중치를 0으로 함. -> 동일한 가중치
        # skew=False -> birth-persistence 축으로 바꾸지 않는다.
        H = pimgr.transform(defect+10, skew=False).flatten() # 100 차원 벡터

        vec = np.concatenate((H0, H1, H)) # 총 300차원 벡터
    else:
        vec = np.concatenate((H0, H1)) # LPI를 제외한 총 200 차원 벡터

    return vec
