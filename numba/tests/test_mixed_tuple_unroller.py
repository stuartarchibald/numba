from __future__ import print_function, absolute_import, division

import numpy as np

from numba.tests.support import TestCase, MemoryLeakMixin
from numba import njit


class TestMixedTupleUnroll(MemoryLeakMixin, TestCase):
    def test_1(self):

        @njit
        def foo(idx, z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for i in range(len(a)):
                acc += a[i]
                if acc.real < 26:
                    acc -= 1
                else:
                    break
            return acc

        f = 9
        k = f

        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_2(self):

        @njit
        def foo(idx, z):
            x = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for a in x:
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    break
            return acc

        f = 9
        k = f

        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_3(self):

        @njit
        def foo(idx, z):
            x = (12, 12.7, 3j, 4, z, 2 * z)
            y = ('foo', z, 2 * z)
            acc = 0
            for a in x:
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    for t in y:
                        print(t)
                    break
            return acc

        f = 9
        k = f

        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_4(self):

        @njit
        def foo(tup):
            acc = 0
            for a in tup:
                acc += a.sum()
            return acc

        n = 10
        tup = (np.ones((n,)), np.ones((n, n)), np.ones((n, n, n)))
        self.assertEqual(foo(tup), foo.py_func(tup))


    def test_5(self):

        @njit
        def foo(tup1, tup2):
            acc = 0
            idx = 0
            for a in tup1:
                if a == 'a':
                    acc += tup2[0].sum()
                elif a == 'b':
                    acc += tup2[1].sum()
                elif a == 'c':
                    acc += tup2[2].sum()
                elif a == 12:
                    acc += tup2[3].sum()
                elif a == 3j:
                    acc += tup2[3].sum()
                elif a == ('f',):
                    acc += tup2[3].sum()
                else:
                    print(a)
                idx +=1
            return acc

        n = 10
        tup1 = ('a','b','c', 12, 3j, ('f',))
        tup2 = (np.ones((n,)), np.ones((n, n)), np.ones((n, n, n)),
                np.ones((n, n, n, n)), np.ones((n, n, n, n, n)))
        self.assertEqual(foo(tup1, tup2), foo.py_func(tup1, tup2))

if __name__ == '__main__':
    unittest.main()
