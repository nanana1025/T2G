import cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt

ctypedef np.float32_t float32
ctypedef np.float64_t float64
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.int32_t int
ctypedef np.int64_t int64

cdef extern from "ucr_dtw.h":
    int ucr_query(float64* q, int m, float64 r, float64* buffer, int buflen, ucr_index* result);
    struct ucr_index:
        float64 value
        int64   index

@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.nonecheck(False)
def query(float64[::1] data not None, float64[::1] query not None, float64 r):
    cdef Py_ssize_t n_data
    cdef Py_ssize_t n_query
    cdef ucr_index result
    cdef int return_code
    n_data = data.shape[0]
    n_query= query.shape[0]
    return_code = ucr_query(&query[0], n_query, r, &data[0], n_data, &result)
    if return_code != 0:
        result.index = -1
    return result


def compute_lc(double[:] data, double[:] query):
    cdef int n = min(data.shape[0], query.shape[0])
    cdef int max_lag = n // 2
    cdef int tau, i
    cdef double sum_lc = 0.0
    cdef int count = 0
    cdef int new_len
    cdef double mean_data, mean_query
    cdef double numerator, var_data, var_query
    cdef double diff_data, diff_query
    cdef double lc

    for tau in range(max_lag + 1):
        if tau >= n:
            continue

        new_len = n - tau
        if new_len < 2:
            continue

        mean_data = 0.0
        mean_query = 0.0

        for i in range(new_len):
            mean_data += data[i + tau]
            mean_query += query[i]

        mean_data /= new_len
        mean_query /= new_len

        numerator = 0.0
        var_data = 0.0
        var_query = 0.0

        for i in range(new_len):
            diff_data = data[i + tau] - mean_data
            diff_query = query[i] - mean_query
            numerator += diff_data * diff_query
            var_data += diff_data * diff_data
            var_query += diff_query * diff_query

        if var_data == 0.0 or var_query == 0.0:
            continue

        lc = numerator / sqrt(var_data * var_query)

        if lc > 0:
            sum_lc += lc
            count += 1

    if count > 0:
        return sum_lc
    else:
        return 0.0

