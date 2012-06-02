# Copyright (C) 2006-2011, Timothy A. Davis.
# Copyright (C) 2012, Richard Lincoln.
# http://www.cise.ufl.edu/research/sparse/CSparse
#
# CSparse.py is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# CSparse.py is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this Module; if not, write to the Free Software
# Foundation, Inc, 51 Franklin St, Fifth Floor, Boston, MA 02110-1301

import time

from os.path import abspath, dirname, join

import unittest

import csparse as cs


class CSparseTest(unittest.TestCase):

    DELTA = 1e-3
    DROP_TOL = 1e-14

    T1 = "t1"

    # Unsymmetric overdetermined pattern of Holland survey. Ashkenazi, 1974
    # http://www.cise.ufl.edu/research/sparse/matrices/HB/ash219.html
    ASH219 = "ash219"

    # Symmetric stiffness matrix small generalized eigenvalue problem.
    # http://www.cise.ufl.edu/research/sparse/matrices/HB/bcsstk01.html
    BCSSTK01 = "bcsstk01"

    # S stiffness matrix - Corp. of Engineers Dam
    # http://www.cise.ufl.edu/research/sparse/matrices/HB/bcsstk16.html
    BCSSTK16 = "bcsstk16"

    # Unsymmetric facsimile convergence matrix.
    # http://www.cise.ufl.edu/research/sparse/matrices/HB/fs_183_1.html
    FS_183_1 = "fs_183_1"

    # Unsymmetric pattern on leaflet advertising ibm 1971 conference,
    # but with the last column removed.
    # http://www.cise.ufl.edu/research/sparse/matrices/HB/ibm32.html
    IBM32A = "ibm32a"

    # The transpose of ibm32a
    IBM32B = "ibm32b"

    # Netlib LP problem afiro: minimize c'*x, where Ax=b, lo<=x<=hi
    # http://www.cise.ufl.edu/research/sparse/matrices/LPnetlib/lp_afiro.html
    LP_AFIRO = "lp_afiro"

    # U Nonsymmetric matrix U.S. Economy 1972 -SZYLD-I.E.A.-NYU-
    # http://www.cise.ufl.edu/research/sparse/matrices/HB/mbeacxc.html
    MBEACXC = "mbeacxc"

    # Cavett problem with 5 components (chemical eng., Westerberg)
    # http://www.cise.ufl.edu/research/sparse/matrices/HB/west0067.html
    WEST0067 = "west0067"

    def get_file(self, name):
        d = abspath(dirname(__file__))
#        return open(join(d, 'matrix', name), 'rb')
        return join(d, 'matrix', name)

    def assert_dimensions(self, A, m, n, nzmax, nnz, norm1=None, delta=1e-3):
        self.assertEquals (m, A.m)
        self.assertEquals (n, A.n)
#        self.assertEquals (nzmax, A.nzmax)

        nz = A.p [A.n] if (A.nz < 0) else A.nz
        self.assertEquals (nnz, nz)
        if norm1 is not None:
            self.assertAlmostEquals (norm1, cs.cs_norm (A), delta=delta)

    def assert_problem(self, prob, m, n, nnz, sym, sym_nnz, norm):
        self.assertEquals (m, prob.A.m)
        self.assertEquals (n, prob.A.n)
        self.assertEquals (nnz, prob.A.p [n])
        self.assertEquals (sym, prob.sym)
        self.assertEquals (sym_nnz, prob.C.p [n] if sym != 0 else 0)
        self.assertEquals (norm, cs.cs_norm (prob.C), 1e-2)

    def assert_structure(self, prob, blocks, singletons, rank):
        self.assertEquals (blocks, prob.nb)
        self.assertEquals (singletons, prob.ns)
        self.assertEquals (rank, prob.sprank)

    def assert_dropped(self, prob, dropped_zeros, dropped_tiny):
        self.assertEquals (dropped_zeros, prob.dropped_zeros)
        self.assertEquals (dropped_tiny, prob.dropped_tiny)

    def is_sym(self, A):
        """1 if A is square & upper tri., -1 if square & lower tri., 0 otherwise
        """
        n = A.n, m = A.m, Ap = A.p, Ai = A.i ;
        if m != n: return (0)
        is_upper = True
        is_lower = True
        for j in range(n):
            for p in range(Ap [j], Ap [j+1]):
                if Ai [p] > j: is_upper = False
                if Ai [p] < j: is_lower = False
        return 1 if is_upper else -1 if is_lower else 0

    def make_sym(self, A):
        """C = A + triu(A,1)'
        """
        AT = cs.cs_transpose (A, True)     # AT = A'
        cs.cs_fkeep (AT, Dropdiag(), None) # drop diagonal entries from AT
        C = cs.cs_add (A, AT, 1, 1)        # C = A+AT
        return C

    def rhs(self, x, b, m):
        """create a right-hand side
        """
        for i in range(m): b[i] = 1 + float(i) / m
        for i in range(m): x[i] = b[i]

    def norm(self, x, n):
        """infinity-norm of x
        """
        normx = 0 ;
        for i in range(n): normx = max(normx, abs(x[i]))
        return normx

    def print_resid(self, ok, A, x, b, resid, prob):
        """compute residual, norm(A*x-b,inf) / (norm(A,1)*norm(x,inf) + norm(b,inf))
        """
        if not ok:
            print "    (failed)\n"
            return

        m = A.m; n = A.n
        for i in range(m):
            resid[i] = -b[i]      # resid = -b
        cs.cs_gaxpy (A, x, resid) # resid = resid + A*x

        r = self.norm (resid, m) / (1 if (n == 0) else (cs.cs_norm (A) * self.norm (x, n) + self.norm (b, m)))
        print "resid: %8.2e" % r

        nrm = self.norm (x, n) ;
        print " (norm: %8.4f, %8.4f)\n" % (nrm, self.norm (b, m))
        prob.norms.append (nrm)

    def tic(self):
        return time.time()

    def toc(self, t):
        s = self.tic()
        return max(0, s - t) / 1000000.0

    def print_order(self, order):
        if order == 0:
            print ("natural    ")
        elif order == 1:
            print ("amd(A+A')  ")
        elif order == 2:
            print ("amd(S'*S)  ")
        elif order == 3:
            print ("amd(A'*A)  ")
        else:
            raise

    def get_problem(self, inp, tol, base=0):
        """Reads a problem from a file.

        @param fileName: file name
        @param tol: drop tolerance
        @param base: file index base
        @return: problem
        """
        prob = Problem()
        T = cs.cs_load (inp, base) # load triplet matrix T from a file
        prob.A = A = cs.cs_compress (T) # A = compressed-column form of T

        if not cs.cs_dupl (A): return None # sum up duplicates
        prob.sym = sym = self.is_sym (A) # determine if A is symmetric
        m = A.m ; n = A.n
        mn = max (m, n)
        nz1 = A.p [n]
        if tol > 0: cs.cs_dropzeros (A) # drop zero entries
        nz2 = A.p [n]
        if tol > 0: cs.cs_droptol (A, tol) # drop tiny entries (just to test)
        prob.C = C = self.make_sym(A) if sym != 0 else A # C = A + triu(A,1)', or C=A
        if C == None: return None
        print ("\n--- Matrix: %d-by-%d, nnz: %d (sym: %d: nnz %d), norm: %8.2e\n" %
            (m, n, A.p [n], sym, C.p [n] if sym != 0 else 0, cs.cs_norm (C)))
        prob.dropped_zeros = nz1 - nz2
        if nz1 != nz2: print "zero entries dropped: %d\n" % nz1 - nz2
        prob.dropped_tiny = nz2 - A.p [n]
        if nz2 != A.p [n]: print "tiny entries dropped: %d\n" % nz2 - A.p [n]
        prob.b = [0.0]*mn
        prob.x = [0.0]*mn
        prob.resid = [0.0]*mn
        return prob


class Problem(object):

    def __init__(self):
        self.A = None
        self.C = None
        self.sym = 0
        self.x = []
        self.b = []
        self.resid = []

        self.norms = []

        self.nb = 0
        self.ns = 0
        self.sprank = 0

        self.dropped_zeros = 0
        self.dropped_tiny = 0


class Dropdiag(cs.cs_ifkeep):
    """true for off-diagonal entries
    """
    def fkeep(self, i, j, aij, other):
        return (i != j)


class CSparseTest1(CSparseTest):
    """Test basic matrix operations.
    """

    def load(self, inp):
        T = cs.cs_load (inp) # load triplet matrix T from file
#        print "T:"
#        cs.cs_print (T, False) # print T
        return T

    def compress(self, T):
        A = cs.cs_compress (T) # A = compressed-column form of T
#        print("A:")
#        cs.cs_print (A, False) # print A
        return A

    def transpose(self, A):
        AT = cs.cs_transpose (A, True) # AT = A'
#        print "AT:"
#        cs.cs_print (AT, False) # print AT
        return AT

    def multiply_add(self, A, AT):
        m = A.m if A != None else 0 # m = # of rows of A
        T = cs.cs_spalloc (m, m, m, True, True) # create triplet identity matrix
        for i in range(m): cs.cs_entry (T, i, i, 1)
        Eye = cs.cs_compress (T) # Eye = speye (m)
        C = cs.cs_multiply (A, AT) # C = A*A'
        D = cs.cs_add(C, Eye, 1, cs.cs_norm (C)) # D = C + Eye*norm (C,1)
#        print "D:"
#        cs.cs_print(D, False) # print D
        return D


    def test_ash219(self):
        fd = self.get_file(CSparseTest.ASH219)

        T = self.load (fd)
        self.assert_dimensions(T, 219, 85, 512, 438)

        A = self.compress(T)
        self.assert_dimensions(A, 219, 85, 438, 438, 9)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 85, 219, 438, 438, 2)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 219, 219, 2205, 2205, 32)


    def test_bcsstk01(self):
        fd = self.get_file(CSparseTest.BCSSTK01)

        T = self.load (fd)
        self.assert_dimensions(T, 48, 48, 256, 224)

        A = self.compress(T)
        self.assert_dimensions(A, 48, 48, 224, 224, 3.00944e+09, 1e4)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 48, 48, 224, 224, 3.57095e+09, 1e4)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 48, 48, 764, 764, 1.73403e+19, 1e14)


    def test_bcsstk16(self):
        fd = self.get_file(CSparseTest.BCSSTK16)

        T = self.load (fd)
        self.assert_dimensions(T, 4884, 4884, 262144, 147631)

        A = self.compress(T)
        self.assert_dimensions(A, 4884, 4884, 147631, 147631, 4.91422e+09, 1e4)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 4884, 4884, 147631, 147631, 5.47522e+09, 1e4)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 4884, 4884, 544856, 544856, 4.13336e+19, 1e14)


    def test_fs_183_1(self):
        fd = self.get_file(CSparseTest.FS_183_1)

        T = self.load (fd)
        self.assert_dimensions(T, 183, 183, 2048, 1069)

        A = self.compress(T)
        self.assert_dimensions(A, 183, 183, 1069, 1069, 1.70318e+09, 1e4)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 183, 183, 1069, 1069, 8.22724e+08, 1e3)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 183, 183, 19665, 19665, 2.80249e+18, 1e13)


    def test_ibm32a(self):
        fd = self.get_file(CSparseTest.IBM32A)

        T = self.load (fd)
        self.assert_dimensions(T, 32, 31, 128, 123)

        A = self.compress(T)
        self.assert_dimensions(A, 32, 31, 123, 123, 7)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 31, 32, 123, 123, 8)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 32, 32, 386, 386, 70)


    def test_ibm32b(self):
        fd = self.get_file(CSparseTest.IBM32B)

        T = self.load (fd)
        self.assert_dimensions(T, 31, 32, 128, 123)

        A = self.compress(T)
        self.assert_dimensions(A, 31, 32, 123, 123, 8)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 32, 31, 123, 123, 7)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 31, 31, 373, 373, 64)


    def test_lp_afiro(self):
        fd = self.get_file(CSparseTest.LP_AFIRO)

        T = self.load (fd)
        self.assert_dimensions(T, 27, 51, 128, 102)

        A = self.compress(T)
        self.assert_dimensions(A, 27, 51, 102, 102, 3.429)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 51, 27, 102, 102, 20.525)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 27, 27, 153, 153, 128.963)


    def test_mbeacxc(self):
        fd = self.get_file(CSparseTest.MBEACXC)

        T = self.load (fd)
        self.assert_dimensions(T, 492, 490, 65536, 49920)

        A = self.compress(T)
        self.assert_dimensions(A, 492, 490, 49920, 49920, 0.928629)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 490, 492, 49920, 49920, 16.5516)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 492, 492, 157350, 157350, 19.6068)


    def test_t1(self):
        fd = self.get_file(CSparseTest.T1)

        T = self.load (fd)
        self.assert_dimensions(T, 4, 4, 16, 10)

        A = self.compress(T)
        self.assert_dimensions(A, 4, 4, 10, 10, 11.1)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 4, 4, 10, 10, 7.7)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 4, 4, 16, 16, 139.58)


    def test_west0067(self):
        fd = self.get_file(CSparseTest.WEST0067)

        T = self.load (fd)
        self.assert_dimensions(T, 67, 67, 512, 299)

        A = self.compress(T)
        self.assert_dimensions(A, 67, 67, 299, 299, 6.14337)

        AT = self.transpose(A)
        self.assert_dimensions(AT, 67, 67, 299, 299, 6.59006)

        D = self.multiply_add(A, AT)
        self.assert_dimensions(D, 67, 67, 1041, 1041, 61.0906)


if __name__ == "__main__":
#    import sys;sys.argv = ['', 'CSparseTest1.test_bcsstk01']
    unittest.main()
