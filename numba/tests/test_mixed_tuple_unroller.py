from __future__ import print_function, absolute_import, division

from numba.tests.support import TestCase, MemoryLeakMixin
from numba import njit


class TestMixedTupleUnroll(MemoryLeakMixin, TestCase):
    def test_foo(self):

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


if __name__ == '__main__':
    unittest.main()
