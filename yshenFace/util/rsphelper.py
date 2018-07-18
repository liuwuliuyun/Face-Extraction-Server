import numpy as np
from json import dumps

def new_c_response(reqId, count):
    rsp_string = ('%s#%d#' % (reqId, count))

    return rsp_string

def new_r_response(reqId, features):
    rsp_string = reqId + '#'
    for feature in features:
        l = len(feature)
        for i in range(l):
            rsp_string = ('%s%f,' % (rsp_string,feature[i]))
        rsp_string = rsp_string + '#'

    
    return rsp_string