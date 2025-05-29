import cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt

def compute_lc(double[:] data, double[:] query, int max_lag=-1):
    cdef int n = min(data.shape[0], query.shape[0])
    cdef int tau, i
    cdef double sum_lc = 0.0
    cdef int count = 0
    cdef int new_len
    cdef double mean_data, mean_query
    cdef double numerator, var_data, var_query
    cdef double diff_data, diff_query
    cdef double lc

    if max_lag == -1:
        max_lag = n // 2  # max_lag 기본값 설정

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

def compute_lc_mean(double[:] data, double[:] query, int max_lag=-1):
    cdef int n = min(data.shape[0], query.shape[0])
    cdef int tau, i
    cdef double sum_lc = 0.0
    cdef int count = 0
    cdef int new_len
    cdef double mean_data, mean_query
    cdef double numerator, var_data, var_query
    cdef double diff_data, diff_query
    cdef double lc

    if max_lag == -1:
        max_lag = n // 2  # max_lag 기본값 설정

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
        return sum_lc /count
    else:
        return 0.0

