# Copyright (C) 2006-2011, Timothy A. Davis.
# Copyright (C) 2012, Richard Lincoln.
# http://www.cise.ufl.edu/research/sparse/CSparse
#
# CSparseJ is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# CSparseJ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this Module; if not, write to the Free Software
# Foundation, Inc, 51 Franklin St, Fifth Floor, Boston, MA 02110-1301

"""PyCSparse: a Python Concise Sparse matrix package

@author: Timothy A. Davis
@author: Richard Lincoln
"""

class cs(object):
    """Matrix in compressed-column or triplet form.
    """
    def __init__(self):
        #: maximum number of entries
        self.nzmax = 0;
        #: number of rows
        self.m = 0
        #: number of columns
        self.n = 0
        #: column pointers (size n+1) or col indices (size nzmax)
        self.p = []
        #: row indices, size nzmax
        self.i = []
        #: numerical values, size nzmax
        self.x = []
        #: # of entries in triplet matrix, -1 for compressed-col
        self.nz = 0


def CS_CSC(A):
    """Returns true if A is in column-compressed form, false otherwise.

    @param A: sparse matrix
    @return: true if A is in column-compressed form, false otherwise
    """
    return A != None and A.nz == -1


def cs_spalloc(m, n, nzmax, values, triplet):
    """Allocate a sparse matrix (triplet form or compressed-column form).

    @param m: number of rows
    @param n: number of columns
    @param nzmax: maximum number of entries
    @param values: allocate pattern only if false, values and pattern otherwise
    @param triplet: compressed-column if false, triplet form otherwise
    @return sparse matrix
    """
    A = cs() # allocate the Dcs object
    A.m, A.n = m, n # define dimensions and nzmax
    A.nzmax = nzmax = max(nzmax, 1)
    A.nz = 0 if triplet else -1 # allocate triplet or comp.col
    A.p = [0]*nzmax if triplet else [0]*n + 1
    A.i = [0]*nzmax
    A.x = [0.0]*nzmax if values else None
    return A


def cs_add(A, B, alpha, beta):
    """C = alpha*A + beta*B

    @param A: column-compressed matrix
    @param B: column-compressed matrix
    @param alpha: scalar alpha
    @param beta: scalar beta
    @return: C=alpha*A + beta*B, null on error
    """
    nz = 0
    if not CS_CSC(A) or not CS_CSC(B):
        return None # check inputs
    if A.m != B.m or A.n != B.n:
        return None
    m, anz, n, Bp, Bx = A.m, A.p[A.n], B.n, B.p, B.x
    bnz = Bp[n]
    w = [0]*m # get workspace
    values = A.x != None and Bx != None
    x = [0.0]*m if values else None # get workspace
    C = cs_spalloc(m, n, anz + bnz, values, False) # allocate result
    Cp, Ci, Cx = C.p, C.i, C.x
    for j in range(n):
        Cp[j] = nz # column j of C starts here
        nz = cs_scatter(A, j, alpha, w, x, j + 1, C, nz) # alpha*A(:,j)
        nz = cs_scatter(B, j, beta, w, x, j + 1, C, nz) # beta*B(:,j)
        if values:
            for p in range(Cp[j], nz):
                Cx[p] = x[Ci[p]];
    Cp[n] = nz # finalize the last column of C
    return C # success; free workspace, return C
