import argparse
import math
import time
from functools import wraps

import numpy as np


def find_atkin_wrapper(method):
    @wraps(method)
    def wrapper(n, batch_size):
        n *= 20
        res = method(n, batch_size)
        return res
    return wrapper


@find_atkin_wrapper
def find_atkin(n, batch_size):
    sz = int(math.sqrt(n))
    x = np.expand_dims(np.arange(1, sz + 1, dtype='int64'), axis=1)
    vals = np.array([], dtype='int64')
    cnt_columns = max(1, int(batch_size / sz))
    for i in range(math.ceil(sz / cnt_columns)):
        sp = i * cnt_columns + 1
        ep = min((i + 1) * cnt_columns + 1, sz + 1)
        y = np.repeat(np.array([np.arange(sp, ep, dtype='int64')]), repeats=sz, axis=0)

        sqr_x, sqr_y = x ** 2, y ** 2

        n1 = 4 * sqr_x + sqr_y
        temp = n1 % 12
        n1_ = n1[(n1 <= n) & ((temp == 1) | (temp == 5))]

        n2 = 3 * sqr_x + sqr_y
        n2_ = n2[(n2 <= n) & (n2 % 12 == 7)]

        n3 = 3 * sqr_x - sqr_y
        n3_ = n3[(n3 <= n) & (n3 % 12 == 11) & (x > y)]

        vals = np.concatenate((n1_, n2_, n3_, vals))

    unique, counts = np.unique(vals, return_counts=True)
    items = unique[counts % 2 == 1]

    data = set(items)

    for x in range(5, sz):
        if x in data:
            for y in range(x ** 2, n + 1, x ** 2):
                if y in data:
                    data.remove(y)

    res = [2, 3] + list(data)
    res.sort()

    return res


def find_eratosthen(n, batch_size):
    res = np.array([], dtype='int32')
    arr = np.arange(2, batch_size, dtype='int32')

    step = 0
    while len(res) < n:
        step += 1
        for i in res:
            arr = arr[(arr % i != 0) | (arr == i)]
            if not len(arr) or i ** 2 > arr[-1]:
                break
        else:
            for i in range(arr.shape[0]):
                item = arr[i]
                if not len(arr) or item ** 2 > arr[-1]:
                    break
                arr = arr[(arr % item != 0) | (arr == item)]

        res = np.concatenate((res, arr))
        arr = np.arange(batch_size * step, batch_size * (step + 1))

    return res


METHODS = {
    'atkin': find_atkin,
    'eratosthen': find_eratosthen,
}
N_LIMITS = {
    'atkin': 50_000_000,
}
BATCH_SIZE = {
    'atkin': 10_000_000,
    'eratosthen': 1_000_000,
}


def main(n, method, batch_size):
    if not isinstance(n, int):
        raise ValueError('Incorrect value of parameter "n"')

    func_solve = METHODS.get(method)
    if func_solve is None:
        raise ValueError(
            f'Incorrect value of parameter "method": Available options:"{[key for key in METHODS.keys()]}"'
        )

    limit_method = N_LIMITS.get(method)
    if limit_method is not None and n > limit_method:
        raise ValueError(f'P parameter value is too high (maximum {limit_method})')

    if batch_size is None:
        batch_size = BATCH_SIZE.get(method)

    res = func_solve(n, batch_size)
    return res[n-1]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', type=int)
    parser.add_argument('-m', '--method', default='eratosthen')
    parser.add_argument('-bs', '--batch_size', type=int, default=None)

    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = get_args()
    t0 = time.time()
    result = main(args.n, args.method, args.batch_size)
    print(f'Answer: {result}')
    print(f'Time: {time.time() - t0:.2f}s')
