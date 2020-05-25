import time
import functools


def timer(func):
    @functools.wraps(func)
    def wrapper_timer():
        st = time.time()
        func()
        et = time.time() - st
        print('Training time: {:.2f}s | {:.2f}min'.format(et, et / 60))
    return wrapper_timer
