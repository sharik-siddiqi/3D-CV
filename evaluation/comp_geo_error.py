import numpy as np

def calc_geo_err(corr, geo_dist):
    errors = np.zeros((geo_dist.shape[0], len(corr)))
    diameter = np.max(geo_dist)
    for idx, i in enumerate(corr):    
        for j in range(errors.shape[0]):
            errors[j,idx] = (geo_dist[j, i[j]] + geo_dist[i[j], j])/2
    return errors/diameter

def comp_all_curve(err, thr):
    c = np.zeros((err.shape[1], thr.shape[0]))
    for i in range(err.shape[1]):
        c[i, :] = calc_err_curve(err[:,i],thr)
    return c

def calc_err_curve(err, thr):
    npoints = err.shape[0]
    curve = np.zeros((1, thr.shape[0]))
    for i in range(thr.shape[0]):
        curve[0,i] = (100*np.sum((err <= thr[i]).astype(int)))/npoints
    return curve
