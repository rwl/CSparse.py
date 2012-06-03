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

"""CSparse.py: a Concise Sparse matrix Python package

@author: Timothy A. Davis
@author: Richard Lincoln
"""

from math import sqrt
from random import randint
from sys import stdout


CS_VER = 1 # CSparse.py Version 1.0.0
CS_SUBVER = 0
CS_SUBSUB = 0
CS_DATE = "May 14, 2012" # CSparse.py release date
CS_COPYRIGHT = "Copyright (C) Timothy A. Davis, 2006-2011"


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


class css(object):
    """Output of symbolic Cholesky, LU, or QR analysis.
    """
    def __init__(self):
        #: inverse row perm. for QR, fill red. perm for Chol
        self.pinv = []
        #: fill-reducing column permutation for LU and QR
        self.q = []
        #: elimination tree for Cholesky and QR
        self.parent = []
        #: column pointers for Cholesky, row counts for QR
        self.cp = []
        #: leftmost[i] = min(find(A(i,:))), for QR
        self.leftmost = []
        #: # of rows for QR, after adding fictitious rows
        self.m2 = 0
        #: # entries in L for LU or Cholesky; in V for QR
        self.lnz = 0
        #: # entries in U for LU; in R for QR
        self.unz = 0


class csn(object):
    """Output of numeric Cholesky, LU, or QR factorization
    """
    def __init__(self):
        #: L for LU and Cholesky, V for QR
        self.L = None
        #: U for LU, R for QR, not used for Cholesky
        self.U = None
        #: partial pivoting for LU
        self.pinv = []
        #: beta [0..n-1] for QR
        self.B = []


class csd(object):
    """Output of Dulmage-Mendelsohn decomposition.
    """
    def __init__(self):
        #: size m, row permutation
        self.p = []
        #: size n, column permutation
        self.q = []
        #: size nb+1, block k is rows r[k] to r[k+1]-1 in A(p,q)
        self.r = []
        #: size nb+1, block k is cols s[k] to s[k+1]-1 in A(p,q)
        self.s = []
        #: # of blocks in fine dmperm decomposition
        self.nb = 0
        #: coarse row decomposition
        self.rr = []
        #: coarse column decomposition
        self.cc = []


def CS_CSC(A):
    """Returns true if A is in column-compressed form, false otherwise.

    @param A: sparse matrix
    @return: true if A is in column-compressed form, false otherwise
    """
    return A != None and A.nz == -1


def CS_TRIPLET(A):
    """Returns true if A is in triplet form, false otherwise.

    @param A: sparse matrix
    @return: true if A is in triplet form, false otherwise
    """
    return A != None and A.nz >= 0


def CS_FLIP(i):
    return -(i) - 2


def CS_UNFLIP(i):
    return CS_FLIP(i) if i < 0 else i


def CS_MARKED(w, j):
    return w[j] < 0


def CS_MARK(w, j):
    w[j] = CS_FLIP(w[j])


class cs_ifkeep(object):
    """Interface for cs_fkeep.
    """
    def fkeep(self, i, j, aij, other):
        """Function used for entries from a sparse matrix.

        @param i: row index
        @param j: column index
        @param aij: value
        @param other: optional parameter
        @return: if false then aij should be dropped
        """
        pass

# Add sparse matrices.

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
    w = ialloc(m) # get workspace
    values = A.x != None and Bx != None
    x = xalloc(m) if values else None # get workspace
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


# Approximate minimum degree ordering.

def _cs_wclear(mark, lemax, w, w_offset, n):
    """clear w
    """
    if mark < 2 or (mark + lemax < 0):
        for k in range(n):
            if w[w_offset + k] != 0:
                w[w_offset + k] = 1
        mark = 2
    return mark # at this point, w [0..n-1] < mark holds


class _cs_diag(cs_ifkeep):
    """keep off-diagonal entries; drop diagonal entries
    """
    def fkeep(self, i, j, aij, other):
        return (i != j)

def cs_amd(order, A):
    """Minimum degree ordering of A+A' (if A is symmetric) or A'A.

    @param order: 0:natural, 1:Chol, 2:LU, 3:QR
    @param A: column-compressed matrix
    @return: amd(A+A') if A is symmetric, or amd(A'A) otherwise, null on error
    or for natural ordering
    """
    nel = 0
    mindeg = 0
    lemax = 0
    # --- Construct matrix C -----------------------------------------------
    if not CS_CSC(A) or order <= 0 or order > 3:
        return None # check
    AT = cs_transpose(A, False) # compute A'
    if AT == None:
        return None
    m, n = A.m, A.n
    dense = max(16, 10 * int(sqrt(n))) # find dense threshold
    dense = min(n - 2, dense)
    if order == 1 and n == m:
        C = cs_add(A, AT, 0, 0) # C = A+A'
    elif order == 2:
        ATp = AT.p # drop dense columns from AT
        ATi = AT.i
        p2 = 0
        for j in range(m):
            p = ATp[j] # column j of AT starts here
            ATp[j] = p2 # new column j starts here
            if ATp[j + 1] - p > dense:
                continue # skip dense col j
            while p < ATp[j + 1]:
                ATi[p2] = ATi[p]
                p2+=1
                p+=1
        ATp[m] = p2 # finalize AT
        A2 = cs_transpose(AT, False) # A2 = AT'
        C = cs_multiply(AT, A2) if A2 != None else None # C=A'*A with no dense rows
        A2 = None
    else:
        C = cs_multiply(AT, A) # C=A'*A
    AT = None
    if C == None:
        return None
    cs_fkeep(C, _cs_diag(), None) # drop diagonal entries
    Cp = C.p
    cnz = Cp[n]
    P = ialloc(n + 1) # allocate result
    W = ialloc(8 * (n + 1)) # get workspace
    t = cnz + cnz / 5 + 2 * n # add elbow room to C
    cs_sprealloc(C, t)
    len = W
    nv = W
    nv_offset = n + 1
    next = W
    next_offset = 2 * (n + 1)
    head = W
    head_offset = 3 * (n + 1)
    elen = W
    elen_offset = 4 * (n + 1)
    degree = W
    degree_offset = 5 * (n + 1)
    w = W
    w_offset = 6 * (n + 1)
    hhead = W
    hhead_offset = 7 * (n + 1)
    last = P # use P as workspace for last
    # --- Initialize quotient graph ----------------------------------------
    for k in range(n):
        len[k] = Cp[k + 1] - Cp[k]
    len[n] = 0
    nzmax = C.nzmax
    Ci = C.i
    for i in range(n):
        head[head_offset + i] = -1 # degree list i is empty
        last[i] = -1
        next[next_offset + i] = -1
        hhead[hhead_offset + i] = -1 # hash list i is empty
        nv[nv_offset + i] = 1 # node i is just one node
        w[w_offset + i] = 1 # node i is alive
        elen[elen_offset + i] = 0 # Ek of node i is empty
        degree[degree_offset + i] = len[i] # degree of node i
    mark = _cs_wclear(0, 0, w, w_offset, n) # clear w
    elen[elen_offset + n] = -2 # n is a dead element
    Cp[n] = -1 # n is a root of assembly tree
    w[w_offset + n] = 0 # n is a dead element
    # --- Initialize degree lists ------------------------------------------
    for i in range(n):
        d = degree[degree_offset + i]
        if d == 0: # node i is empty
            elen[elen_offset + i] = -2 # element i is dead
            nel+=1
            Cp[i] = -1 # i is a root of assembly tree
            w[w_offset + i] = 0
        elif d > dense: # node i is dense
            nv[nv_offset + i] = 0 # absorb i into element n
            elen[elen_offset + i] = -1 # node i is dead
            nel+=1
            Cp[i] = CS_FLIP(n)
            nv[nv_offset + n]+=1
        else:
            if head[head_offset + d] != -1:
                last[head[head_offset + d]] = i
            next[next_offset + i] = head[head_offset + d] # put node i in degree list d
            head[head_offset + d] = i

    while nel < n: # while (selecting pivots) do

        # --- Select node of minimum approximate degree --------------------
        k = -1
        while mindeg < n and k == -1:
            k = head[head_offset + mindeg]
            mindeg += 1
        if next[next_offset + k] != -1:
            last[next[next_offset + k]] = -1
        head[head_offset + mindeg] = next[next_offset + k] # remove k from degree list
        elenk = elen[elen_offset + k] # elenk = |Ek|
        nvk = nv[nv_offset + k] # # of nodes k represents
        nel += nvk # nv[nv_offset+k] nodes of A eliminated
        # --- Garbage collection -------------------------------------------
        if elenk > 0 and cnz + mindeg >= nzmax:
            for j in range(n):
                p = Cp[j]
                if p >= 0: # j is a live node or element
                    Cp[j] = Ci[p] # save first entry of object
                    Ci[p] = CS_FLIP(j) # first entry is now CS_FLIP(j)

            q = 0; p = 0
            while p < cnz: # scan all of memory
                j = CS_FLIP(Ci[p])
                p+=1
                if j >= 0: # found object j
                    Ci[q] = Cp[j] # restore first entry of object
                    Cp[j] = q # new pointer to object j
                    q+=1
                    for k3 in range(len[j] - 1):
                        Ci[q] = Ci[p]
                        q+=1; p+=1
            cnz = q # Ci [cnz...nzmax-1] now free

        # --- Construct new element ----------------------------------------
        dk = 0
        nv[nv_offset + k] = -nvk # flag k as in Lk
        p = Cp[k]
        pk1 = p if elenk == 0 else cnz # do in place if elen[elen_offset+k] == 0
        pk2 = pk1
        for k1 in range(1, elenk + 2):
            if k1 > elenk:
                e = k # search the nodes in k
                pj = p # list of nodes starts at Ci[pj]
                ln = len[k] - elenk # length of list of nodes in k
            else:
                e = Ci[p] # search the nodes in e
                p+=1
                pj = Cp[e]
                ln = len[e] # length of list of nodes in e
            for k2 in range(1, ln + 1):
                i = Ci[pj]
                pj+=1
                nvi = nv[nv_offset + i]
                if nvi <= 0:
                    continue # node i dead, or seen
                dk += nvi # degree[degree_offset+Lk] += size of node i
                nv[nv_offset + i] = -nvi # negate nv[nv_offset+i] to denote i in Lk
                Ci[pk2] = i # place i in Lk
                pk2+=1
                if next[next_offset + i] != -1:
                    last[next[next_offset + i]] = last[i]
                if last[i] != -1: # remove i from degree list
                    next[next_offset + last[i]] = next[next_offset + i]
                else:
                    head[head_offset + degree[degree_offset + i]] = next[next_offset + i]
            if e != k:
                Cp[e] = CS_FLIP(k) # absorb e into k
                w[w_offset + e] = 0 # e is now a dead element
        if elenk != 0:
            cnz = pk2 # Ci [cnz...nzmax] is free
        degree[degree_offset + k] = dk # external degree of k - |Lk\i|
        Cp[k] = pk1 # element k is in Ci[pk1..pk2-1]
        len[k] = pk2 - pk1
        elen[elen_offset + k] = -2 # k is now an element
        # --- Find set differences -----------------------------------------
        mark = _cs_wclear(mark, lemax, w, w_offset, n) # clear w if necessary
        for pk in range(pk1,  pk2): # scan 1: find |Le\Lk|
            i = Ci[pk]
            eln = elen[elen_offset + i]
            if eln <= 0:
                continue # skip if elen[elen_offset+i] empty
            nvi = -nv[nv_offset + i] # nv [i] was negated
            wnvi = mark - nvi
            p = Cp[i]
            while p <= Cp[i] + eln - 1: # scan Ei
                e = Ci[p]
                if w[w_offset + e] >= mark:
                    w[w_offset + e] -= nvi # decrement |Le\Lk|
                elif w[w_offset + e] != 0: # ensure e is a live element
                    w[w_offset + e] = degree[degree_offset + e] + wnvi # 1st time e seen in scan 1
                p += 1
        # --- Degree update ------------------------------------------------
        for pk in range(pk2): # scan2: degree update
            i = Ci[pk] # consider node i in Lk
            p1 = Cp[i]
            p2 = p1 + elen[elen_offset + i] - 1
            pn = p1
            h = 0; d = 0; p = p1
            while p <= p2: # scan Ei
                e = Ci[p]
                if w[w_offset + e] != 0: # e is an unabsorbed element
                    dext = w[w_offset + e] - mark # dext = |Le\Lk|
                    if dext > 0:
                        d += dext # sum up the set differences
                        Ci[pn] = e # keep e in Ei
                        pn+=1
                        h += e # compute the hash of node i
                    else:
                        Cp[e] = CS_FLIP(k) # aggressive absorb. e.k
                        w[w_offset + e] = 0 # e is a dead element
                p+=1
            elen[elen_offset + i] = pn - p1 + 1 # elen[elen_offset+i] = |Ei|
            p3 = pn
            p4 = p1 + len[i]
            for p in range(p2 + 1, p < p4): # prune edges in Ai
                j = Ci[p]
                nvj = nv[nv_offset + j]
                if nvj <= 0:
                    continue # node j dead or in Lk
                d += nvj # degree(i) += |j|
                Ci[pn] = j # place j in node list of i
                pn+=1
                h += j # compute hash for node i
            if d == 0: # check for mass elimination
                Cp[i] = CS_FLIP(k) # absorb i into k
                nvi = -nv[nv_offset + i]
                dk -= nvi # |Lk| -= |i|
                nvk += nvi # |k| += nv[nv_offset+i]
                nel += nvi
                nv[nv_offset + i] = 0
                elen[elen_offset + i] = -1 # node i is dead
            else:
                degree[degree_offset + i] = min(degree[degree_offset + i], d) # update degree(i)
                Ci[pn] = Ci[p3] # move first node to end
                Ci[p3] = Ci[p1] # move 1st el. to end of Ei
                Ci[p1] = k # add k as 1st element in of Ei
                len[i] = pn - p1 + 1 # new len of adj. list of node i
                h %= n # finalize hash of i
                next[next_offset + i] = hhead[hhead_offset + h] # place i in hash bucket
                hhead[hhead_offset + h] = i
                last[i] = h # save hash of i in last[i]
        # scan2 is done
        degree[degree_offset + k] = dk # finalize |Lk|
        lemax = max(lemax, dk)
        mark = _cs_wclear(mark + lemax, lemax, w, w_offset, n) # clear w
        # --- Supernode detection ------------------------------------------
        for pk in range(pk1, pk2):
            i = Ci[pk]
            if nv[nv_offset + i] >= 0:
                continue # skip if i is dead
            h = last[i] # scan hash bucket of node i
            i = hhead[hhead_offset + h]
            hhead[hhead_offset + h] = -1 # hash bucket will be empty
            while i != -1 and next[next_offset + i] != -1:
                ln = len[i]
                eln = elen[elen_offset + i]
                p = Cp[i] + 1
                while p <= Cp[i] + ln - 1:
                    w[w_offset + Ci[p]] = mark
                    p+=1
                jlast = i
                j = next[next_offset + i]
                while j != -1: # compare i with all j
                    ok = (len[j] == ln) and (elen[elen_offset + j] == eln)
                    p = Cp[j] + 1
                    while ok and p <= Cp[j] + ln - 1:
                        if w[w_offset + Ci[p]] != mark:
                            ok = False # compare i and j
                        p+=1
                    if ok: # i and j are identical
                        Cp[j] = CS_FLIP(i) # absorb j into i
                        nv[nv_offset + i] += nv[nv_offset + j]
                        nv[nv_offset + j] = 0
                        elen[elen_offset + j] = -1 # node j is dead
                        j = next[next_offset + j] # delete j from hash bucket
                        next[next_offset + jlast] = j
                    else:
                        jlast = j # j and i are different
                        j = next[next_offset + j]
                i = next[next_offset + i]; mark+=1
        # --- Finalize new element------------------------------------------
        p = pk1
        for pk in range(pk1, pk2): # finalize Lk
            i = Ci[pk]
            nvi = -nv[nv_offset + i]
            if nvi <= 0:
                continue # skip if i is dead
            nv[nv_offset + i] = nvi # restore nv[nv_offset+i]
            d = degree[degree_offset + i] + dk - nvi # compute external degree(i)
            d = min(d, n - nel - nvi)
            if head[head_offset + d] != -1:
                last[head[head_offset + d]] = i
            next[next_offset + i] = head[head_offset + d] # put i back in degree list
            last[i] = -1
            head[head_offset + d] = i
            mindeg = min(mindeg, d) # find new minimum degree
            degree[degree_offset + i] = d
            Ci[p] = i # place i in Lk
            p+=1
        nv[nv_offset + k] = nvk # # nodes absorbed into k
        len[k] = p - pk1
        if len[k] == 0: # length of adj list of element k
            Cp[k] = -1 # k is a root of the tree
            w[w_offset + k] = 0 # k is now a dead element
        if elenk != 0:
            cnz = p # free unused space in Lk
    # --- Postordering -----------------------------------------------------
    for i in range(n):
        Cp[i] = CS_FLIP(Cp[i]) # fix assembly tree
    j = 0
    while j <= n:
        head[head_offset + j] = -1
        j+=1
    j = n
    while j >= 0: # place unordered nodes in lists
        if nv[nv_offset + j] > 0:
            j-=1
            continue # skip if j is an element
        next[next_offset + j] = head[head_offset + Cp[j]] # place j in list of its parent
        head[head_offset + Cp[j]] = j
        j-=1
    e = n
    while e >= 0: # place elements in lists
        if nv[nv_offset + e] <= 0:
            e-=1
            continue # skip unless e is an element
        if Cp[e] != -1:
            next[next_offset + e] = head[head_offset + Cp[e]] # place e in list of its parent
            head[head_offset + Cp[e]] = e
        e-=1
    k = 0; i = 0
    while i <= n: # postorder the assembly tree
        if Cp[i] == -1:
            k = cs_tdfs(i, k, head, head_offset, next, next_offset, P, 0, w, w_offset)
        i+=1
    return P


# Sparse Cholesky.

def cs_chol(A, S):
    """Numeric Cholesky factorization LL=PAP'.

    @param A: column-compressed matrix, only upper triangular part is used
    @param S: symbolic Cholesky analysis, pinv is optional
    @return: numeric Cholesky factorization, null on error
    """
    if not CS_CSC(A) or S == None or S.cp == None or S.parent == None:
        return None
    n = A.n
    N = csn() # allocate result
    c = ialloc(2 * n) # get int workspace
    x = xalloc(n) # get double workspace
    cp = S.cp
    pinv = S.pinv
    parent = S.parent
    C = cs_symperm(A, pinv, True) if pinv != None else A
    s = c
    s_offset = n
    Cp = C.p
    Ci = C.i
    Cx = C.x
    N.L = L = cs_spalloc(n, n, cp[n], True, False) # allocate result
    Lp = L.p
    Li = L.i
    Lx = L.x
    for k in range(n):
        Lp[k] = c[k] = cp[k]
    for k in range(n): # compute L(k,:) for L*L' = C
        # --- Nonzero pattern of L(k,:) ------------------------------------
        top = cs_ereach(C, k, parent, s, s_offset, c) # find pattern of L(k,:)
        x[k] = 0 # x (0:k) is now zero
        for p in range(Cp[k], p < Cp[k + 1]): # x = full(triu(C(:,k)))
            if Ci[p] <= k:
                x[Ci[p]] = Cx[p]
        d = x[k] # d = C(k,k)
        x[k] = 0 # clear x for k+1st iteration
        # --- Triangular solve ---------------------------------------------
        while top < n: # solve L(0:k-1,0:k-1) * x = C(:,k)
            i = s[s_offset + top] # s [top..n-1] is pattern of L(k,:)
            lki = x[i] / Lx[Lp[i]] # L(k,i) = x (i) / L(i,i)
            x[i] = 0 # clear x for k+1st iteration
            for p in range(Lp[i] + 1, p < c[i]):
                x[Li[p]] -= Lx[p] * lki
            d -= lki * lki # d = d - L(k,i)*L(k,i)
            p = c[i]
            c[i]+=1
            Li[p] = k # store L(k,i) in column i
            Lx[p] = lki
            top+=1
        # --- Compute L(k,k) -----------------------------------------------
        if d <= 0:
            return None # not pos def
        p = c[k]
        c[k]+=1
        Li[p] = k # store L(k,k) = sqrt (d) in column k
        Lx[p] = sqrt(d)
    Lp[n] = cp[n] # finalize L
    return N


def cs_cholsol(order, A, b):
    """Solves Ax=b where A is symmetric positive definite; b is overwritten
    with solution.

    @param order: ordering method to use (0 or 1)
    @param A: column-compressed matrix, symmetric positive definite, only
              upper triangular part is used
    @param b: right hand side, b is overwritten with solution
    @return: true if successful, false on error
    """
    if not CS_CSC(A) or b == None:
        return False # check inputs
    n = A.n
    S = cs_schol(order, A) # ordering and symbolic analysis
    N = cs_chol(A, S) # numeric Cholesky factorization
    x = xalloc(n) # get workspace
    ok = S != None and N != None
    if ok:
        cs_ipvec(S.pinv, b, x, n) # x = P*b
        cs_lsolve(N.L, x) # x = L\x
        cs_ltsolve(N.L, x) # x = L'\x
        cs_pvec(S.pinv, x, b, n) # b = P'*x
    return ok


def cs_compress(T):
    """C = compressed-column form of a triplet matrix T. The columns of C are
    not sorted, and duplicate entries may be present in C.

    @param T: triplet matrix
    @return: C if successful, null on error
    """
    if not CS_TRIPLET(T):
        return None # check inputs
    m, n = T.m, T.n
    Ti, Tj, Tx, nz = T.i, T.p, T.x, T.nz
    C = cs_spalloc(m, n, nz, Tx != None, False) # allocate result
    w = ialloc(n) # get workspace
    Cp = C.p
    Ci = C.i
    Cx = C.x
    for k in range(nz):
        w[Tj[k]]+=1 # column counts
    cs_cumsum(Cp, w, n) # column pointers
    for k in range(nz):
        p = w[Tj[k]]
        w[Tj[k]]+=1
        Ci[p] = Ti[k] # A(i,j) is the pth entry in C
        if Cx != None:
            Cx[p] = Tx[k]
    return C


# Column counts for Cholesky and QR.

def _HEAD(k, j, head, head_offset, ata):
    return head[head_offset + k] if ata else j


def _NEXT(J, next, next_offset, ata):
    return next[next_offset + J] if ata else -1


def _init_ata(AT, post, w):
    m = AT.n; n = AT.m; ATp = AT.p; ATi = AT.i
    head = w
    head_offset = 4 * n
    next = w
    next_offset = 5 * n + 1
    for k in range(n):
        w[post[k]] = k # invert post
    for i in range(m):
        k = n
        for p in range(ATp[i], ATp[i + 1]):
            k = min(k, w[ATi[p]])
        next[next_offset + i] = head[head_offset + k] # place row i in linked list k
        head[head_offset + k] = i

    return head_offset, next_offset


def cs_counts(A, parent, post, ata):
    """Column counts of LL'=A or LL'=A'A, given parent & postordering

    @param A: column-compressed matrix
    @param parent: elimination tree of A
    @param post: postordering of parent
    @param ata: analyze A if false, A'A otherwise
    @return column counts of LL'=A or LL'=A'A, null on error
    """
    jleaf = [0]
    if not CS_CSC(A) or parent == None or post == None:
        return None # check inputs
    m, n = A.m, A.n
    s = 4 * n + (n + m + 1 if ata else 0)
    delta = colcount = ialloc(n) # allocate result
    w = ialloc(s) # get workspace
    AT = cs_transpose(A, False) # AT = A'
    ancestor = w
    maxfirst = w
    maxfirst_offset = n
    prevleaf = w
    prevleaf_offset = 2 * n
    first = w
    first_offset = 3 * n
    for k in range(s):
        w[k] = -1 # clear workspace w [0..s-1]
    for k in range(n): # find first [j]
        j = post[k]
        delta[j] = 1 if first[first_offset + j] == -1 else 0 # delta[j]=1 if j is a leaf
        while j != -1 and first[first_offset + j] == -1:
            first[first_offset + j] = k
            j = parent[j]
    ATp, ATi = AT.p, AT.i
    if ata:
        offsets = _init_ata(AT, post, w)
        head = w
        head_offset = offsets[0]
        next = w
        next_offset = offsets[1]
    for i in range(n):
        ancestor[i] = i # each node in its own set
    for k in range(n):
        j = post[k] # j is the kth node in postordered etree
        if parent[j] != -1:
            delta[parent[j]]-=1 # j is not a root
        J = _HEAD(k, j, head, head_offset, ata)
        while J != -1: # J=j for LL'=A case
            for p in range(ATp[J], ATp[J + 1]):
                i = ATi[p]
                q = cs_leaf(i, j, first, first_offset, maxfirst, maxfirst_offset,
                        prevleaf, prevleaf_offset, ancestor, 0, jleaf)
                if jleaf[0] >= 1:
                    delta[j]+=1 # A(i,j) is in skeleton
                if jleaf[0] == 2:
                    delta[q]-=1 # account for overlap in q
            J = _NEXT(J, next, next_offset, ata)
        if parent[j] != -1:
            ancestor[j] = parent[j]
    for j in range(n): # sum up delta's of each child
        if parent[j] != -1:
            colcount[parent[j]] += colcount[j]
    return colcount


def cs_cumsum(p, c, n):
    """p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c

    @param p: size n+1, cumulative sum of c
    @param c: size n, overwritten with p [0..n-1] on output
    @param n: length of c
    @return: sum (c), null on error
    """
    nz = 0
    nz2 = 0.0
    if p == None or c == None: return -1 # check inputs
    for i in range(n):
        p[i] = nz
        nz += c[i]
        nz2 += c[i]              # also in double to avoid CS_INT overflow
        c[i] = p[i]             # also copy p[0..n-1] back into c[0..n-1]
    p[n] = nz
    return int(nz2)               # return sum (c [0..n-1])


# Depth-first-search.

def cs_dfs(j, G, top, xi, xi_offset, pstack, pstack_offset, pinv, pinv_offset):
    """Depth-first-search of the graph of a matrix, starting at node j.

    @param j: starting node
    @param G: graph to search (G.p modified, then restored)
    @param top: stack[top..n-1] is used on input
    @param xi: size n, stack containing nodes traversed
    @param xi_offset: the index of the first element in array xi
    @param pstack: size n, work array
    @param pstack_offset: the index of the first element in array pstack
    @param pinv: mapping of rows to columns of G, ignored if null
    @param pinv_offset: the index of the first element in array pinv
    @return new value of top, -1 on error
    """
    head = 0
    if not CS_CSC(G) or xi == None or pstack == None:
        return -1 # check inputs
    Gp, Gi = G.p, G.i
    xi[xi_offset + 0] = j # initialize the recursion stack
    while head >= 0:
        j = xi[xi_offset + head] # get j from the top of the recursion stack
        jnew = pinv[pinv_offset + j] if pinv != None else j
        if not CS_MARKED(Gp, j):
            CS_MARK(Gp, j) # mark node j as visited
            pstack[pstack_offset + head] = 0 if jnew < 0 else CS_UNFLIP(Gp[jnew])
        done = True # node j done if no unvisited neighbors
        p2 = 0 if jnew < 0 else CS_UNFLIP(Gp[jnew + 1])
        for p in range(pstack[pstack_offset + head], p2): # examine all neighbors of j
            i = Gi[p] # consider neighbor node i
            if CS_MARKED(Gp, i):
                continue # skip visited node i
            pstack[pstack_offset + head] = p # pause depth-first search of node j
            head+=1
            xi[xi_offset + head] = i # start dfs at node i
            done = False # node j is not done
            break # break, to start dfs (i)
        if done: # depth-first search at node j is done
            head-=1 # remove j from the recursion stack
            top-=1
            xi[xi_offset + top] = j # and place in the output stack
    return top


# Dulmage-Mendelsohn decomposition.

def _cs_bfs(A, n, wi, wj, queue, imatch, imatch_offset, jmatch, jmatch_offset, mark):
    """breadth-first search for coarse decomposition (C0,C1,R1 or R0,R3,C3)
    """
    head = 0; tail = 0
    for j in range(n): # place all unmatched nodes in queue
        if imatch[imatch_offset + j] >= 0:
            continue # skip j if matched
        wj[j] = 0 # j in set C0 (R0 if transpose)
        queue[tail] = j # place unmatched col j in queue
        tail+=1
    if tail == 0:
        return True # quick return if no unmatched nodes
    C = A if mark == 1 else cs_transpose(A, False)
    if C == None:
        return False # bfs of C=A' to find R3,C3 from R0
    Ap, Ai = C.p, C.i
    while head < tail: # while queue is not empty
        j = queue[head] # get the head of the queue
        head+=1
        for p in range(Ap[j], Ap[j + 1]):
            i = Ai[p]
            if wi[i] >= 0:
                continue # skip if i is marked
            wi[i] = mark # i in set R1 (C3 if transpose)
            j2 = jmatch[jmatch_offset + i] # traverse alternating path to j2
            if wj[j2] >= 0:
                continue # skip j2 if it is marked
            wj[j2] = mark # j2 in set C1 (R3 if transpose)
            queue[tail] = j2 # add j2 to queue
            tail+=1
    if mark != 1:
        C = None # free A' if it was created
    return True


def _cs_matched(n, wj, imatch, imatch_offset, p, q, cc, rr, set, mark):
    """collect matched rows and columns into p and q
    """
    kc = cc[set]
    kr = rr[set - 1]
    for j in range(n):
        if wj[j] != mark:
            continue # skip if j is not in C set
        p[kr] = imatch[imatch_offset + j]
        kr+=1
        q[kc] = j
        kc+=1
    cc[set + 1] = kc
    rr[set] = kr


def _cs_unmatched(m, wi, p, rr, set):
    """collect unmatched rows into the permutation vector p
    """
    kr = rr[set]
    for i in range(m):
        if wi[i] == 0:
            p[kr] = i
            kr+=1
    rr[set + 1] = kr


class _cs_rprune(cs_ifkeep):
    """return 1 if row i is in R2
    """

    def fkeep(self, i, j, aij, other):
        rr = other
        return i >= rr[1] and i < rr[2]


def cs_dmperm(A, seed):
    """Compute coarse and then fine Dulmage-Mendelsohn decomposition. seed
    optionally selects a randomized algorithm.

    @param A: column-compressed matrix
    @param seed: 0: natural, -1: reverse, random order oterwise
    @return: Dulmage-Mendelsohn analysis, null on error
    """
    # --- Maximum matching -------------------------------------------------
    if not CS_CSC(A):
        return None # check inputs
    m, n = A.m, A.n
    D = cs_dalloc(m, n) # allocate result
    if D == None:
        return None
    p, q, r, s, cc, rr = D.p, D.q, D.r, D.s, D.cc, D.rr
    jmatch = cs_maxtrans(A, seed) # max transversal
    imatch = jmatch # imatch = inverse of jmatch
    imatch_offset = m
    if jmatch == None:
        return None
    # --- Coarse decomposition ---------------------------------------------
    wi = r
    wj = s # use r and s as workspace
    for j in range(n):
        wj[j] = -1 # unmark all cols for bfs
    for i in range(m):
        wi[i] = -1 # unmark all rows for bfs
    _cs_bfs(A, n, wi, wj, q, imatch, imatch_offset, jmatch, 0, 1) # find C1, R1 from C0
    ok = _cs_bfs(A, m, wj, wi, p, jmatch, 0, imatch, imatch_offset, 3) # find R3, C3 from R0
    if not ok:
        return None
    _cs_unmatched(n, wj, q, cc, 0) # unmatched set C0
    _cs_matched(n, wj, imatch, imatch_offset, p, q, cc, rr, 1, 1) # set R1 and C1
    _cs_matched(n, wj, imatch, imatch_offset, p, q, cc, rr, 2, -1) # set R2 and C2
    _cs_matched(n, wj, imatch, imatch_offset, p, q, cc, rr, 3, 3) # set R3 and C3
    _cs_unmatched(m, wi, p, rr, 3) # unmatched set R0
    jmatch = None
    # --- Fine decomposition -----------------------------------------------
    pinv = cs_pinv(p, m) # pinv=p'
    if pinv == None:
        return None
    C = cs_permute(A, pinv, q, False) # C=A(p,q) (it will hold A(R2,C2))
    pinv = None
    if C == None:
        return None
    Cp = C.p
    nc = cc[3] - cc[2] # delete cols C0, C1, and C3 from C
    if cc[2] > 0:
        j = cc[2]
        while j <= cc[3]:
            Cp[j - cc[2]] = Cp[j]
            j+=1
    C.n = nc
    if rr[2] - rr[1] < m: # delete rows R0, R1, and R3 from C
        cs_fkeep(C, _cs_rprune(), rr)
        cnz = Cp[nc]
        Ci = C.i
        if rr[1] > 0:
            for k in range(cnz):
                Ci[k] -= rr[1]
    C.m = nc
    scc = cs_scc(C) # find strongly connected components of C
    if scc == None:
        return None
    # --- Combine coarse and fine decompositions ---------------------------
    ps = scc.p # C(ps,ps) is the permuted matrix
    rs = scc.r # kth block is rs[k]..rs[k+1]-1
    nb1 = scc.nb # # of blocks of A(R2,C2)
    for k in range(nc):
        wj[k] = q[ps[k] + cc[2]]
    for k in range(nc):
        q[k + cc[2]] = wj[k]
    for k in range(nc):
        wi[k] = p[ps[k] + rr[1]]
    for k in range(nc):
        p[k + rr[1]] = wi[k]
    nb2 = 0 # create the fine block partitions
    r[0] = s[0] = 0
    if cc[2] > 0:
        nb2+=1 # leading coarse block A (R1, [C0 C1])
    for k in range(nb1): # coarse block A (R2,C2)
        r[nb2] = rs[k] + rr[1] # A (R2,C2) splits into nb1 fine blocks
        s[nb2] = rs[k] + cc[2]
        nb2+=1
    if rr[2] < m:
        r[nb2] = rr[2] # trailing coarse block A ([R3 R0], C3)
        s[nb2] = cc[3]
        nb2+=1
    r[nb2] = m
    s[nb2] = n
    D.nb = nb2
    return D


# Drop small entries from a sparse matrix.

class _cs_tol(cs_ifkeep):
    def fkeep(self, i, j, aij, other):
        return abs(aij) > float(other)


def cs_droptol(A, tol):
    """Removes entries from a matrix with absolute value <= tol.

    @param A: column-compressed matrix
    @param tol: drop tolerance
    @return: nz, new number of entries in A, -1 on error
    """
    return cs_fkeep(A, _cs_tol(), tol) # keep all large entries


# Drop zeros from a sparse matrix.

class _cs_nonzero(cs_ifkeep):
    def fkeep(self, i, j, aij, other):
        return aij != 0


def cs_dropzeros(A):
    """Removes numerically zero entries from a matrix.

    @param A: column-compressed matrix
    @return: nz, new number of entries in A, -1 on error
    """
    return cs_fkeep(A, _cs_nonzero(), None) # keep all nonzero entries


# Remove (and sum) duplicates.

def cs_dupl(A):
    """Removes and sums duplicate entries in a sparse matrix.
     *
     * @param A
     *            column-compressed matrix
     * @return true if successful, false on error
    """
    nz = 0
    if not CS_CSC(A): # check inputs
        return False
    m, n = A.m, A.n
    Ap, Ai, Ax = A.p, A.i, A.x
    w = ialloc(m) # get workspace
    for i in range(m):
        w[i] = -1 # row i not yet seen
    for j in range(n):
        q = nz # column j will start at q
        for p in range(Ap[j], Ap[j + 1]):
            i = Ai[p] # A(i,j) is nonzero
            if w[i] >= q:
                Ax[w[i]] += Ax[p] # A(i,j) is a duplicate
            else:
                w[i] = nz # record where row i occurs
                Ai[nz] = i # keep A(i,j)
                Ax[nz] = Ax[p]
                nz+=1
        Ap[j] = q # record start of column j
    Ap[n] = nz # finalize A
    return cs_sprealloc(A, 0) # remove extra space from A


# Add an entry to a triplet matrix.

def cs_entry(T, i, j, x):
    """Adds an entry to a triplet matrix. Memory-space and dimension of T are
    increased if necessary.

    @param T: triplet matrix; new entry added on output
    @param i: row index of new entry
    @param j: column index of new entry
    @param x: numerical value of new entry
    @return: true if successful, false otherwise
    """
    if not CS_TRIPLET(T) or i < 0 or j < 0:
        return False # check inputs
    if T.nz >= T.nzmax:
        cs_sprealloc(T, 2 * (T.nzmax))
    if T.x != None:
        T.x[T.nz] = x
    T.i[T.nz] = i
    T.p[T.nz] = j
    T.nz+=1
    T.m = max(T.m, i + 1)
    T.n = max(T.n, j + 1)
    return True


# Nonzero pattern of kth row of Cholesky factor, L(k,1:k-1).

def cs_ereach(A, k, parent, s, s_offset, w):
    """Find nonzero pattern of Cholesky L(k,1:k-1) using etree and triu(A(:,k)).
    If ok, s[top..n-1] contains pattern of L(k,:).

    @param A: column-compressed matrix; L is the Cholesky factor of A
    @param k: find kth row of L
    @param parent: elimination tree of A
    @param s: size n, s[top..n-1] is nonzero pattern of L(k,1:k-1)
    @param s_offset: the index of the first element in array s
    @param w: size n, work array, w[0..n-1]>=0 on input, unchanged on output

    @return top in successful, -1 on error
    """
#    int i, p, n, len, top, Ap[], Ai[];
    if not CS_CSC(A) or parent == None or s == None or w == None:
        return -1 # check inputs
    top = n = A.n
    Ap = A.p
    Ai = A.i
    CS_MARK(w, k) # mark node k as visited
    for p in range(Ap[k], Ap[k + 1]):
        i = Ai[p] # A(i,k) is nonzero
        if i > k:
            continue # only use upper triangular part of A
        len = 0
        while not CS_MARKED(w, i): # traverse up etree
            s[s_offset + len] = i # L(k,i) is nonzero
            len+=1
            CS_MARK(w, i) # mark i as visited
            i = parent[i]
        while len > 0:
            top-=1
            len-=1
            s[s_offset + top] = s[s_offset + len] # push path onto stack
    for p in range(top, n):
        CS_MARK(w, s[s_offset + p]) # unmark all nodes
    CS_MARK(w, k) # unmark node k
    return top # s [top..n-1] contains pattern of L(k,:)


# Find elimination tree.

def cs_etree(A, ata):
    """Compute the elimination tree of A or A'A (without forming A'A).

    @param A: column-compressed matrix
    @param ata: analyze A if false, A'A oterwise
    @return: elimination tree, null on error
    """
#    int i, k, p, m, n, inext, Ap[], Ai[], w[], parent[], ancestor[], prev[];
    if not CS_CSC(A):
        return None # check inputs
    m, n = A.m, A.n
    Ap, Ai = A.p, A.i
    parent = ialloc(n) # allocate result
    w = ialloc(n + (m if ata else 0)) # get workspace
    ancestor = w
    prev = w
    prev_offset = n
    if ata:
        for i in range(m):
            prev[prev_offset + i] = -1
    for k in range(n):
        parent[k] = -1 # node k has no parent yet
        ancestor[k] = -1 # nor does k have an ancestor
        for p in range(Ap[k], Ap[k + 1]):
            i = prev[prev_offset + Ai[p]] if ata else Ai[p]
            while i != -1 and i < k: # traverse from i to k
                inext = ancestor[i] # inext = ancestor of i
                ancestor[i] = k # path compression
                if inext == -1:
                    parent[i] = k # no anc., parent is k
                i = inext
            if ata:
                prev[prev_offset + Ai[p]] = k
    return parent


def cs_fkeep(A, fkeep, other):
    """Drops entries from a sparse matrix;

    @param A: column-compressed matrix
    @param fkeep: drop aij if fkeep.fkeep(i,j,aij,other) is false
    @param other: optional parameter to fkeep
    @return: nz, new number of entries in A, -1 on error
    """
    nz = 0
    if not CS_CSC(A):
        return (-1) # check inputs
    n, Ap, Ai, Ax = A.n, A.p, A.i, A.x
    for j in range(n):
        p = Ap[j] # get current location of col j
        Ap[j] = nz # record new location of col j
        while p < Ap[j + 1]:
            if fkeep.fkeep(Ai[p], j, Ax[p] if Ax != None else 1, other):
                if Ax != None:
                    Ax[nz] = Ax[p] # keep A(i,j)
                Ai[nz] = Ai[p]
                nz+=1
            p+=1
    Ap[n] = nz # finalize A
    cs_sprealloc(A, 0) # remove extra space from A
    return nz


def cs_gaxpy(A, x, y):
    """Sparse matrix times dense column vector, y = A*x+y.

    @param A: column-compressed matrix
    @param x: size n, vector x
    @param y: size m, vector y
    @return: true if successful, false on error
    """
    if not CS_CSC(A) or x == None or y == None:
        return False # check inputs
    n, Ap, Ai, Ax = A.n, A.p, A.i, A.x
    for j in range(n):
        for p in range(Ap[j], Ap[j + 1]):
            y[Ai[p]] += Ax[p] * x[j]
    return True


def cs_happly(V, i, beta, x):
    """Applies a Householder reflection to a dense vector,
    x = (I - beta*v*v')*x.

    @param V: column-compressed matrix of Householder vectors
    @param i: v = V(:,i), the ith column of V
    @param beta: scalar beta
    @param x: vector x of size m
    @return true if successful, false on error
    """
    tau = 0;
    if not CS_CSC(V) or x == None:
        return False # check inputs
    Vp, Vi, Vx = V.p, V.i, V.x
    for p in range(Vp[i], Vp[i + 1]): # tau = v'*x
        tau += Vx[p] * x[Vi[p]]
    tau *= beta # tau = beta*(v'*x)
    for p in range(Vp[i], Vp[i + 1]): # x = x - v*tau
        x[Vi[p]] -= Vx[p] * tau
    return True


def cs_house(x, x_offset, beta, n):
    """Compute a Householder reflection, overwrite x with v, where
    (I-beta*v*v')*x = s*e1. See Algo 5.1.1, Golub & Van Loan, 3rd ed.

    @param x: x on output, v on input
    @param x_offset: the index of the first element in array x
    @param beta: scalar beta
    @param n: the length of x
    @return: norm2(x), -1 on error
    """
    sigma = 0
    if x == None or beta == None:
        return -1 # check inputs
    for i in range(1, n):
        sigma += x[x_offset + i] * x[x_offset + i]
    if sigma == 0:
        s = abs(x[x_offset + 0]) # s = |x(0)|
        beta[0] = 2.0 if x[x_offset + 0] <= 0 else 0.0
        x[x_offset + 0] = 1
    else:
        s = sqrt(x[x_offset + 0] * x[x_offset + 0] + sigma) # s = norm (x)
        x[x_offset + 0] = x[x_offset + 0] - s if x[x_offset + 0] <= 0 else -sigma / (x[x_offset + 0] + s)
        beta[0] = -1.0 / (s * x[x_offset + 0])
    return s


def cs_ipvec(p, b, x, n):
    """Permutes a vector, x = P'b.

    @param p: permutation vector, p=null denotes identity
    @param b: input vector
    @param x: output vector, x = P'b
    @param n: length of p, b, and x
    @return: true if successful, false on error
    """
    if x == None or b == None:
        return False # check inputs
    for k in range(n):
        x[p[k] if p != None else k] = b[k]
    return True


def cs_leaf(i, j, first, first_offset, maxfirst, maxfirst_offset,
        prevleaf, prevleaf_offset, ancestor, ancestor_offset, jleaf):
    """Determines if j is a leaf of the skeleton matrix and find lowest common
    ancestor (lca).
    """
    if first == None or maxfirst == None or prevleaf == None or ancestor == None or jleaf == None:
        return -1
    jleaf[0] = 0
    if i <= j or first[first_offset + j] <= maxfirst[maxfirst_offset + i]:
        return -1 # j not a leaf
    maxfirst[maxfirst_offset + i] = first[first_offset + j] # update max first[j] seen so far
    jprev = prevleaf[prevleaf_offset + i] # jprev = previous leaf of ith subtree
    prevleaf[prevleaf_offset + i] = j
    jleaf[0] = 1 if jprev == -1 else 2 # j is first or subsequent leaf
    if jleaf[0] == 1:
        return i # if 1st leaf, q = root of ith subtree
    q = jprev
    while q != ancestor[ancestor_offset + q]:
        q = ancestor[ancestor_offset + q]
    s = jprev
    while s != q:
        sparent = ancestor[ancestor_offset + s] # path compression
        ancestor[ancestor_offset + s] = q
        s = sparent
    return q # q = least common ancestor (jprev,j)


def cs_load(filename, base=0):
    """Loads a triplet matrix T from a file. Each line of the file contains
    three values: a row index i, a column index j, and a numerical value aij.
    The file is zero-based.

    @param filename: file name
    @param base: index base
    @return: T if successful, null on error
    """
    T = cs_spalloc(0, 0, 1, True, True) # allocate result
    with open(filename, 'rb') as fd:
        for line in fd:
            tokens = line.strip().split()
            if len(tokens) != 3:
                return None
            i = int(tokens[0]) - base
            j = int(tokens[1]) - base
            x = float(tokens[2])
            if not cs_entry(T, i, j, x):
                return None
    return T


def cs_lsolve(L, x):
    """Solves a lower triangular system Lx=b where x and b are dense. x=b on
    input, solution on output.

    @param L: column-compressed, lower triangular matrix
    @param x: size n, right hand side on input, solution on output
    @return: true if successful, false on error
    """
    if not CS_CSC(L) or x == None:
        return False # check inputs
    n, Lp, Li, Lx = L.n, L.p, L.i, L.x
    for j in range(n):
        x[j] /= Lx[Lp[j]]
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[Li[p]] -= Lx[p] * x[j]
    return True


def cs_ltsolve(L, x):
    """Solves an upper triangular system L'x=b where x and b are dense. x=b on
    input, solution on output.

    @param L: column-compressed, lower triangular matrix
    @param x: size n, right hand side on input, solution on output
    @return true if successful, false on error
    """
    if not CS_CSC(L) or x == None:
        return False # check inputs
    n, Lp, Li, Lx = L.n, L.p, L.i, L.x
    j = n - 1
    while j >= 0:
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[j] -= Lx[p] * x[Li[p]]
        x[j] /= Lx[Lp[j]]
        j-=1
    return True


# Sparse LU factorization.

def cs_lu(A, S, tol):
    """Sparse LU factorization of a square matrix, PAQ = LU.

    @param A: column-compressed matrix
    @param S: symbolic LU analysis
    @param tol: partial pivoting threshold (1 for partial pivoting)
    @return: numeric LU factorization, null on error
    """
    if not CS_CSC(A) or S == None:
        return None # check inputs
    n, q = A.n, S.q
    lnz = S.lnz
    unz = S.unz
    x = xalloc(n) # get double workspace
    xi = ialloc(2 * n) # get int workspace
    N = csn() # allocate result
    N.L = L = cs_spalloc(n, n, lnz, True, False) # allocate result L
    N.U = U = cs_spalloc(n, n, unz, True, False) # allocate result U
    N.pinv = pinv = ialloc(n) # allocate result pinv
    Lp = L.p
    Up = U.p
    for i in range(n):
        x[i] = 0 # clear workspace
    for i in range(n):
        pinv[i] = -1 # no rows pivotal yet
    for k in range(n+1):
        Lp[k] = 0 # no cols of L yet
    lnz = unz = 0
    for k in range(n): # compute L(:,k) and U(:,k)
        # --- Triangular solve ---------------------------------------------
        Lp[k] = lnz # L(:,k) starts here
        Up[k] = unz # U(:,k) starts here
        if lnz + n > L.nzmax:
            cs_sprealloc(L, 2 * L.nzmax + n)
        if unz + n > U.nzmax:
            cs_sprealloc(U, 2 * U.nzmax + n)
        Li, Lx, Ui, Ux = L.i, L.x, U.i, U.x
        col = q[k] if q != None else k
        top = cs_spsolve(L, A, col, xi, x, pinv, True) # x = L\A(:,col)
        # --- Find pivot ---------------------------------------------------
        ipiv = -1
        a = -1
        for p in range(top, n):
            i = xi[p] # x(i) is nonzero
            if pinv[i] < 0: # row i is not yet pivotal
                t = abs(x[i])
                if t > a:
                    a = t # largest pivot candidate so far
                    ipiv = i
            else: # x(i) is the entry U(pinv[i],k)
                Ui[unz] = pinv[i]
                Ux[unz] = x[i]
                unz+=1
        if ipiv == -1 or a <= 0:
            return None
        if pinv[col] < 0 and abs(x[col]) >= a * tol:
            ipiv = col
        # --- Divide by pivot ----------------------------------------------
        pivot = x[ipiv] # the chosen pivot
        Ui[unz] = k # last entry in U(:,k) is U(k,k)
        Ux[unz] = pivot
        unz+=1
        pinv[ipiv] = k # ipiv is the kth pivot row
        Li[lnz] = ipiv # first entry in L(:,k) is L(k,k) = 1
        Lx[lnz] = 1
        lnz+=1
        for p in range(n): # L(k+1:n,k) = x / pivot
            i = xi[p]
            if pinv[i] < 0: # x(i) is an entry in L(:,k)
                Li[lnz] = i # save unpermuted row in L
                Lx[lnz] = x[i] / pivot # scale pivot column
                lnz+=1
            x[i] = 0 # x [0..n-1] = 0 for next k
    # --- Finalize L and U -------------------------------------------------
    Lp[n] = lnz
    Up[n] = unz
    Li = L.i # fix row indices of L for final pinv
    for p in range(lnz):
        Li[p] = pinv[Li[p]]
    cs_sprealloc(L, 0) # remove extra space from L and U
    cs_sprealloc(U, 0)
    return N


# Solve Ax=b using sparse LU factorization.

def cs_lusol(order, A, b, tol):
    """Solves Ax=b, where A is square and nonsingular. b overwritten with
    solution. Partial pivoting if tol = 1.

    @param order: ordering method to use (0 to 3)
    @param A: column-compressed matrix
    @param b: size n, b on input, x on output
    @param tol: partial pivoting tolerance
    @return: true if successful, false on error
    """
    if not CS_CSC(A) or b == None:
        return False # check inputs
    n = A.n
    S = cs_sqr(order, A, False) # ordering and symbolic analysis
    N = cs_lu(A, S, tol) # numeric LU factorization
    x = xalloc(n) # get workspace
    ok = S != None and N != None
    if ok:
        cs_ipvec(N.pinv, b, x, n) # x = b(p)
        cs_lsolve(N.L, x) # x = L\x
        cs_usolve(N.U, x) # x = U\x
        cs_ipvec(S.q, x, b, n) # b(q) = x
    return ok


# Maximum transveral (permutation for zero-free diagonal).

def _cs_augment(k, A, jmatch, jmatch_offset, cheap, cheap_offset,
        w, w_offset, js, js_offset, is_, is_offset, ps, ps_offset):
    """find an augmenting path starting at column k and extend the match if found
    """
    head = 0
    i = -1
    Ap,Ai = A.p, A.i
    found = False
    js[js_offset + 0] = k # start with just node k in jstack
    while head >= 0:
        # --- Start (or continue) depth-first-search at node j -------------
        j = js[js_offset + head] # get j from top of jstack
        if w[w_offset + j] != k: # 1st time j visited for kth path
            w[w_offset + j] = k # mark j as visited for kth path
            p = cheap[cheap_offset + j]
            while p < Ap[j + 1] and found:
                i = Ai[p] # try a cheap assignment (i,j)
                found = jmatch[jmatch_offset + i] == -1
                p+=1
            cheap[cheap_offset + j] = p # start here next time j is traversed
            if found:
                is_[is_offset + head] = i # column j matched with row i
                break # end of augmenting path
            ps[ps_offset + head] = Ap[j] # no cheap match: start dfs for j
        # --- Depth-first-search of neighbors of j -------------------------
        for p in range(ps[ps_offset + head], Ap[j + 1]):
            i = Ai[p] # consider row i
            if w[w_offset + jmatch[jmatch_offset + i]] == k:
                continue # skip jmatch [i] if marked
            ps[ps_offset + head] = p + 1 # pause dfs of node j
            is_[is_offset + head] = i # i will be matched with j if found
            head+=1
            js[js_offset + head] = jmatch[jmatch_offset + i] # start dfs at column jmatch [i]
            break
        if p == Ap[j + 1]:
            head-=1 # node j is done; pop from stack
    # augment the match if path found:
    if found:
        p = head
        while p >= 0:
            jmatch[jmatch_offset + is_[is_offset + p]] = js[js_offset + p]
            p-=1


def cs_maxtrans(A, seed): # [jmatch [0..m-1]; imatch [0..n-1]]
    """Find a maximum transveral (zero-free diagonal). Seed optionally selects
    a randomized algorithm.

    @param A: column-compressed matrix
    @param seed: 0: natural, -1: reverse, randomized otherwise
    @return row and column matching, size m+n
    """
    n2 = m2 = 0
    if not CS_CSC(A):
        return None # check inputs
    n, m, Ap, Ai = A.n, A.m, A.p, A.i
    w = jimatch = ialloc(m + n) # allocate result
    k = 0
    for j in range(n): # count nonempty rows and columns
        if Ap[j] < Ap[j + 1]:
            n2+=1
        for p in range(Ap[j], Ap[j + 1]):
            w[Ai[p]] = 1
            if j == Ai[p]:
                k+=1 # count entries already on diagonal
    if k == min(m, n): # quick return if diagonal zero-free
        jmatch = jimatch
        imatch = jimatch
        imatch_offset = m
        for i in range(k):
            jmatch[i] = i
        while i < m:
            jmatch[i] = -1
            i+=1
        for j in range(k):
            imatch[imatch_offset + j] = j
        while j < n:
            imatch[imatch_offset + j] = -1
            j+=1
        return jimatch
    for i in range(m):
        m2 += w[i]
    C = cs_transpose(A, False) if m2 < n2 else A # transpose if needed
    if C == None:
        return None
    n, m, Cp = C.n, C.m, C.p
    jmatch = jimatch
    imatch = jimatch
    jmatch_offset = 0
    imatch_offset = 0
    if m2 < n2:
        jmatch_offset = n
    else:
        imatch_offset = m
    w = ialloc(5 * n) # get workspace
    cheap = w
    cheap_offset = n
    js = w
    js_offset = 2 * n
    is_ = w
    is_offset = 3 * n
    ps = w
    ps_offset = 4 * n
    for j in range(n):
        cheap[cheap_offset + j] = Cp[j] # for cheap assignment
    for j in range(n):
        w[j] = -1 # all columns unflagged
    for i in range(m):
        jmatch[jmatch_offset + i] = -1 # nothing matched yet
    q = cs_randperm(n, seed) # q = random permutation
    for k in range(n): # augment, starting at column q[k]
        _cs_augment(q[k] if q != None else k, C, jmatch, jmatch_offset,
                    cheap, cheap_offset, w, 0, js, js_offset,
                    is_, is_offset, ps, ps_offset)
    q = None
    for j in range(n):
        imatch[imatch_offset + j] = -1 # find row match
    for i in range(m):
        if jmatch[jmatch_offset + i] >= 0:
            imatch[imatch_offset + jmatch[jmatch_offset + i]] = i
    return jimatch


# Sparse matrix multiply.

def cs_multiply(A, B):
    """Sparse matrix multiplication, C = A*B

    @param A: column-compressed matrix
    @param B: column-compressed matrix
    @return: C = A*B, null on error
    """
    nz = 0
    if not CS_CSC(A) or not CS_CSC(B):
        return None # check inputs
    if A.n != B.m:
        return None
    m = A.m
    anz = A.p[A.n]
    n, Bp, Bi, Bx = B.n, B.p, B.i, B.x
    bnz = Bp[n]
    w = ialloc(m) # get workspace
    values = (A.x != None) and (Bx != None)
    x = xalloc(m) if values else None # get workspace
    C = cs_spalloc(m, n, anz + bnz, values, False) # allocate result
    Cp = C.p
    for j in range(n):
        if nz + m > C.nzmax:
            cs_sprealloc(C, 2 * (C.nzmax) + m)
        Ci = C.i
        Cx = C.x # C.i and C.x may be reallocated
        Cp[j] = nz # column j of C starts here
        for p in range(Bp[j], Bp[j + 1]):
            nz = cs_scatter(A, Bi[p], Bx[p] if Bx != None else 1, w, x, j + 1, C, nz)
        if values:
            for p in range(Cp[j], nz):
                Cx[p] = x[Ci[p]]
    Cp[n] = nz # finalize the last column of C
    cs_sprealloc(C, 0) # remove extra space from C
    return C


# Sparse matrix 1-norm.

def cs_norm(A):
    """Computes the 1-norm of a sparse matrix = max (sum (abs (A))), largest
    column sum.

    @param A: column-compressed matrix
    @return: the 1-norm if successful, -1 on error
    """
    norm = 0
    if not CS_CSC(A) or A.x == None:
        return -1 # check inputs
    n, Ap, Ax = A.n, A.p, A.x
    for j in range(n):
        s = 0
        for p in range(Ap[j], Ap[j + 1]):
            s += abs(Ax[p])
        norm = max(norm, s)
    return norm


def cs_permute(A, pinv, q, values):
    """Permutes a sparse matrix, C = PAQ.

    @param A: m-by-n, column-compressed matrix
    @param pinv: a permutation vector of length m
    @param q: a permutation vector of length n
    @param values: allocate pattern only if false, values and pattern otherwise
    @return: C = PAQ, null on error
    """
    nz = 0
#    int t, j, k, , m, n, Ap[], Ai[], Cp[], Ci[];
#    double Cx[], Ax[];
#    Dcs C;
    if not CS_CSC(A):
        return None # check inputs
    m, n, Ap, Ai, Ax = A.m, A.n, A.p, A.i, A.x
    C = cs_spalloc(m, n, Ap[n], values and Ax != None, False) # alloc result
    Cp, Ci, Cx = C.p, C.i, C.x
    for k in range(n):
        Cp[k] = nz # column k of C is column q[k] of A
        j = q[k] if q != None else k
        for t in range(Ap[j], Ap[j + 1]):
            if Cx != None:
                Cx[nz] = Ax[t] # row i of A is row pinv[i] of C
            Ci[nz] = pinv[Ai[t]] if pinv != None else Ai[t]
            nz+=1
    Cp[n] = nz # finalize the last column of C
    return C


def cs_pinv(p, n):
    """Inverts a permutation vector. Returns pinv[i] = k if p[k] = i on input.

    @param p: a permutation vector if length n
    @param n: length of p
    @return: pinv, null on error
    """
    if p == None:
        return None # p = NULL denotes identity
    pinv = ialloc(n) # allocate result
    for k in range(n):
        pinv[p[k]] = k # invert the permutation
    return pinv # return result


def cs_post(parent, n):
    """Postorders a tree of forest.

    @param parent: defines a tree of n nodes
    @param n: length of parent
    @return: post[k]=i, null on error
    """
    k = 0
    if parent == None:
        return None # check inputs
    post = ialloc(n) # allocate result
    w = ialloc(3 * n) # get workspace
    head = w
    next = w
    next_offset = n
    stack = w
    stack_offset = 2 * n
    for j in range(n):
        head[j] = -1 # empty linked lists
    j = n - 1
    while j >= 0: # traverse nodes in reverse order
        if parent[j] == -1:
            j-=1
            continue # j is a root
        next[next_offset + j] = head[parent[j]] # add j to list of its parent
        head[parent[j]] = j
        j-=1
    for j in range(n):
        if parent[j] != -1:
            continue # skip j if it is not a root
        k = cs_tdfs(j, k, head, 0, next, next_offset, post, 0, stack, stack_offset)
    return post


def cs_print(A, brief):
    """Prints a sparse matrix.

    @param A: sparse matrix (triplet ot column-compressed)
    @param brief: print all of A if false, a few entries otherwise
    @return: true if successful, false on error
    """
    if A == None:
        stdout.write("(null)\n")
        return False
    m, n, Ap, Ai, Ax = A.m, A.n, A.p, A.i, A.x
    nzmax = A.nzmax
    nz = A.nz
    stdout.write("CSparse.py Version %d.%d.%d, %s.  %s\n" % (CS_VER, CS_SUBVER,
            CS_SUBSUB, CS_DATE, CS_COPYRIGHT))
    if nz < 0:
        stdout.write("%d-by-%d, nzmax: %d nnz: %d, 1-norm: %g\n" % (m, n, nzmax, Ap[n], cs_norm(A)))
        for j in range(n):
            stdout.write("    col %d : locations %d to %d\n" % (j, Ap[j], Ap[j + 1] - 1))
            for p in range(Ap[j], Ap[j + 1]):
                stdout.write("      %d : %g\n" % (Ai[p], Ax[p] if Ax != None else 1))
                if brief and p > 20:
                    stdout.write("  ...\n")
                    return True
    else:
        stdout.write("triplet: %d-by-%d, nzmax: %d nnz: %d\n" % (m, n, nzmax, nz))
        for p in range(nz):
            stdout.write("    %d %d : %g\n" % (Ai[p], Ap[p], Ax[p] if Ax != None else 1))
            if brief and p > 20:
                stdout.write("  ...\n")
                return True
    return True


def cs_pvec(p, b, x, n):
    """Permutes a vector, x=P*b, for dense vectors x and b.

    @param p: permutation vector, p=null denotes identity
    @param b: input vector
    @param x: output vector, x=P*b
    @param n: length of p, b and x
    @return: true if successful, false otherwise
    """
    if x == None or b == None:
        return False # check inputs
    for k in range(n):
        x[k] = b[p[k] if p != None else k]
    return True


# Sparse QR factorization.

def cs_qr(A, S):
    """Sparse QR factorization of an m-by-n matrix A, A= Q*R

    @param A: column-compressed matrix
    @param S: symbolic QR analysis
    @return: numeric QR factorization, null on error
    """
    if not CS_CSC(A) or S == None:
        return None
    n, Ap, Ai, Ax = A.n, A.p, A.i, A.x
    q, parent, pinv, m2, vnz, rnz, leftmost = S.q, S.parent, S.pinv, S.m2, S.lnz, S.unz, S.leftmost
    w = ialloc(m2 + n) # get int workspace
    x = xalloc(m2) # get double workspace
    N = csn() # allocate result
    s = w
    s_offset = m2 # s is size n
    for k in range(m2):
        x[k] = 0 # clear workspace x
    N.L = V = cs_spalloc(m2, n, vnz, True, False) # allocate result V
    N.U = R = cs_spalloc(m2, n, rnz, True, False) # allocate result R
    N.B = Beta = xalloc(n) # allocate result Beta
    Rp, Ri, Rx = R.p, R.i, R.x
    Vp, Vi, Vx = V.p, V.i, V.x
    for i in range(m2):
        w[i] = -1 # clear w, to mark nodes
    rnz = 0
    vnz = 0
    for k in range(n): # compute V and R
        Rp[k] = rnz # R(:,k) starts here
        Vp[k] = p1 = vnz # V(:,k) starts here
        w[k] = k # add V(k,k) to pattern of V
        Vi[vnz] = k
        vnz+=1
        top = n
        col = q[k] if q != None else k
        for p in range(Ap[col], Ap[col + 1]): # find R(:,k) pattern
            i = leftmost[Ai[p]] # i = min(find(A(i,q)))
            len = 0
            while w[i] != k: # traverse up to k
                s[s_offset + len] = i
                len+=1
                w[i] = k
                i = parent[i]
            while len > 0:
                top-=1
                len-=1
                s[s_offset + top] = s[s_offset + len] # push path on stack
            i = pinv[Ai[p]] # i = permuted row of A(:,col)
            x[i] = Ax[p] # x (i) = A(:,col)
            if i > k and w[i] < k: # pattern of V(:,k) = x (k+1:m)
                Vi[vnz] = i # add i to pattern of V(:,k)
                vnz+=1
                w[i] = k
        for p in range(top, n): # for each i in pattern of R(:,k)
            i = s[s_offset + p] # R(i,k) is nonzero
            cs_happly(V, i, Beta[i], x) # apply (V(i),Beta(i)) to x
            Ri[rnz] = i # R(i,k) = x(i)
            Rx[rnz] = x[i]
            rnz+=1
            x[i] = 0
            if parent[i] == k:
                vnz = cs_scatter(V, i, 0, w, None, k, V, vnz)
        for p in range(p1, vnz): # gather V(:,k) = x
            Vx[p] = x[Vi[p]]
            x[Vi[p]] = 0
        Ri[rnz] = k # R(k,k) = norm (x)
        beta = [0.0]
        beta[0] = Beta[k]
        Rx[rnz] = cs_house(Vx, p1, beta, vnz - p1) # [v,beta]=house(x)
        rnz+=1
        Beta[k] = beta[0]
    Rp[n] = rnz # finalize R
    Vp[n] = vnz # finalize V
    return N


# Solve a least-squares or underdetermined problem.

def cs_qrsol(order, A, b):
    """Solve a least-squares problem (min ||Ax-b||_2, where A is m-by-n with
    m >= n) or underdetermined system (Ax=b, where m < n)

    @param order: ordering method to use (0 to 3)
    @param A: column-compressed matrix
    @param b: size max(m,n), b (size m) on input, x(size n) on output
    @return: true if successful, false on error
    """
    if not CS_CSC(A) or b == None:
        return False # check inputs
    n, m = A.n, A.m
    if m >= n:
        S = cs_sqr(order, A, True) # ordering and symbolic analysis
        N = cs_qr(A, S) # numeric QR factorization
        x = xalloc(S.m2 if S != None else 1) # get workspace
        ok = S != None and N != None
        if ok:
            cs_ipvec(S.pinv, b, x, m) # x(0:m-1) = b(p(0:m-1)
            for k in range(n): # apply Householder refl. to x
                cs_happly(N.L, k, N.B[k], x)
            cs_usolve(N.U, x) # x = R\x
            cs_ipvec(S.q, x, b, n) # b(q(0:n-1)) = x(0:n-1)
    else:
        AT = cs_transpose(A, True) # Ax=b is underdetermined
        S = cs_sqr(order, AT, True) # ordering and symbolic analysis
        N = cs_qr(AT, S) # numeric QR factorization of A'
        x = xalloc(S.m2 if S != None else 1) # get workspace
        ok = AT != None and S != None and N != None
        if ok:
            cs_pvec(S.q, b, x, m) # x(q(0:m-1)) = b(0:m-1)
            cs_utsolve(N.U, x) # x = R'\x
            k = m - 1
            while k >= 0: # apply Householder refl. to x
                cs_happly(N.L, k, N.B[k], x)
                k-=1
            cs_pvec(S.pinv, x, b, n) # b(0:n-1) = x(p(0:n-1))
    return ok


def cs_randperm(n, seed):
    """Returns a random permutation vector, the identity perm, or p = n-1:-1:0.
    seed = -1 means p = n-1:-1:0. seed = 0 means p = identity. otherwise p =
    random permutation.

    @param n: length of p
    @param seed: 0: natural, -1: reverse, random p oterwise
    @return: p, null on error or for natural order
    """
    if seed == 0:
        return None # return p = NULL (identity)
    p = ialloc(n) # allocate result
    for k in range(n):
        p[k] = n - k - 1
    if seed == -1:
        return p # return reverse permutation
    for k in range(n):
        j = k + randint(n - k) # j = rand int in range k to n-1
        t = p[j] # swap p[k] and p[j]
        p[j] = p[k]
        p[k] = t
    return p


def cs_reach(G, B, k, xi, pinv):
    """Finds a nonzero pattern of x=L\b for sparse L and b.

    @param G: graph to search (G.p modified, then restored)
    @param B: right hand side, b = B(:,k)
    @param k: use kth column of B
    @param xi: size 2*n, output in xi[top..n-1]
    @param pinv: mapping of rows to columns of G, ignored if null
    @return: top, -1 on error
    """
    if not CS_CSC(G) or not CS_CSC(B) or xi == None:
        return -1 # check inputs
    n, Bp, Bi, Gp = G.n, B.p, B.i, G.p
    top = n
    for p in range(Bp[k], Bp[k + 1]):
        if not CS_MARKED(Gp, Bi[p]): # start a dfs at unmarked node i
            top = cs_dfs(Bi[p], G, top, xi, 0, xi, n, pinv, 0)
    for p in range(top, n):
        CS_MARK(Gp, xi[p]) #restore G
    return top


def cs_scatter(A, j, beta, w, x, mark, C, nz):
    """Scatters and sums a sparse vector A(:,j) into a dense vector, x = x +
    beta * A(:,j).

    @param A: the sparse vector is A(:,j)
    @param j: the column of A to use
    @param beta: scalar multiplied by A(:,j)
    @param w: size m, node i is marked if w[i] = mark
    @param x: size m, ignored if null
    @param mark: mark value of w
    @param C: pattern of x accumulated in C.i
    @param nz: pattern of x placed in C starting at C.i[nz]
    @return: new value of nz, -1 on error
    """
    if not CS_CSC(A) or w == None or not CS_CSC(C):
        return -1 # check inputs
    Ap, Ai, Ax = A.p, A.i, A.x
    Ci = C.i
    for p in range(Ap[j], Ap[j + 1]):
        i = Ai[p] # A(i,j) is nonzero
        if w[i] < mark:
            w[i] = mark # i is new entry in column j
            Ci[nz] = i # add i to pattern of C(:,j)
            nz+=1
            if x != None:
                x[i] = beta * Ax[p] # x(i) = beta*A(i,j)
        elif x != None:
            x[i] += beta * Ax[p] # i exists in C(:,j) already
    return nz


def cs_scc(A):
    """Finds the strongly connected components of a square matrix.

    @param A: column-compressed matrix (A.p modified then restored)
    @return: strongly connected components, null on error
    """
    if not CS_CSC(A):
        return None # check inputs
    n = A.n
    Ap = A.p
    D = cs_dalloc(n, 0) # allocate result
    AT = cs_transpose(A, False) # AT = A'
    xi = ialloc(2 * n + 1) # get workspace
    if D == None or AT == None:
        return None
    Blk = xi
    rcopy = xi
    rcopy_offset = n
    pstack = xi
    pstack_offset = n
    p = D.p
    r = D.r
    ATp = AT.p
    top = n
    for i in range(n): # first dfs(A) to find finish times (xi)
        if not CS_MARKED(Ap, i):
            top = cs_dfs(i, A, top, xi, 0, pstack, pstack_offset, None, 0)
    for i in range(n):
        CS_MARK(Ap, i) # restore A; unmark all nodes
    top = n
    nb = n
    for k in range(n): # dfs(A') to find strongly connnected comp
        i = xi[k] # get i in reverse order of finish times
        if CS_MARKED(ATp, i):
            continue # skip node i if already ordered
        r[nb] = top # node i is the start of a component in p
        nb-=1
        top = cs_dfs(i, AT, top, p, 0, pstack, pstack_offset, None, 0)
    r[nb] = 0 # first block starts at zero; shift r up
    k = nb
    while k <= n:
        r[k - nb] = r[k]
        k+=1
    D.nb = nb = n - nb # nb = # of strongly connected components
    for b in range(nb): # sort each block in natural order
        for k in range(r[b], r[b + 1]):
            Blk[p[k]] = b
    b = 0
    while b <= nb:
        rcopy[rcopy_offset + b] = r[b];
        b+=1
    for i in range(n):
        p[rcopy[rcopy_offset + Blk[i]]] = i
        rcopy[rcopy_offset + Blk[i]]+=1
    return D


# Symbolic Cholesky ordering and analysis.

def cs_schol(order, A):
    """Ordering and symbolic analysis for a Cholesky factorization.

    @param order: ordering option (0 or 1)
    @param A: column-compressed matrix
    @return: symbolic analysis for Cholesky, null on error
    """
    if not CS_CSC(A):
        return None  # check inputs
    n = A.n
    S = css() # allocate result S
    P = cs_amd(order, A) # P = amd(A+A'), or natural
    S.pinv = cs_pinv(P, n) # find inverse permutation
    if order != 0 and S.pinv == None:
        return None
    C = cs_symperm(A, S.pinv, False) # C = spones(triu(A(P,P)))
    S.parent = cs_etree(C, False) # find etree of C
    post = cs_post(S.parent, n) # postorder the etree
    c = cs_counts(C, S.parent, post, False) # find column counts of chol(C)
    S.cp = ialloc(n + 1) # allocate result S.cp
    S.unz = S.lnz = cs_cumsum(S.cp, c, n) # find column pointers for L
    return S if S.lnz >= 0 else None


# Sparse lower or upper triangular solve. x=G\b where G, x, and b are sparse,
# and G upper/lower triangular.

def cs_spsolve(G, B, k, xi, x, pinv, lo):
    """Solve Gx=b(:,k), where G is either upper (lo=false) or lower (lo=true)
    triangular.

    @param G: lower or upper triangular matrix in column-compressed form
    @param B: right hand side, b=B(:,k)
    @param k: use kth column of B as right hand side
    @param xi: size 2*n, nonzero pattern of x in xi[top..n-1]
    @param x: size n, x in x[xi[top..n-1]]
    @param pinv: mapping of rows to columns of G, ignored if null
    @param lo: true if lower triangular, false if upper
    @return: top, -1 in error
    """
#    int j, J, p, q, px, top, n, Gp[], Gi[], Bp[], Bi[];
#    double Gx[], Bx[];
    if not CS_CSC(G) or not CS_CSC(B) or xi == None or x == None:
        return -1
    Gp, Gi, Gx, n = G.p, G.i, G.x, G.n
    Bp, Bi, Bx = B.p, B.i, B.x
    top = cs_reach(G, B, k, xi, pinv) # xi[top..n-1]=Reach(B(:,k))
    for p in range(top, n):
        x[xi[p]] = 0 # clear x
    for p in range(Bp[k], Bp[k + 1]):
        x[Bi[p]] = Bx[p] # scatter B
    for px in range(top, n):
        j = xi[px] # x(j) is nonzero
        J = pinv[j] if pinv != None else j # j maps to col J of G
        if J < 0:
            continue # column J is empty
        x[j] /= Gx[Gp[J] if lo else Gp[J + 1] - 1] # x(j) /= G(j,j)
        p = Gp[J] + 1 if lo else Gp[J] # lo: L(j,j) 1st entry
        q = Gp[J + 1] if lo else Gp[J + 1] - 1 # up: U(j,j) last entry
        while p < q:
            x[Gi[p]] -= Gx[p] * x[j] # x(i) -= G(i,j) * x(j)
            p+=1
    return top # return top of stack


# Symbolic QR or LU ordering and analysis.

def _cs_vcount(A, S):
    """compute nnz(V) = S->lnz, S->pinv, S->leftmost, S->m2 from A and S->parent
    """
    n = A.n; m = A.m; Ap = A.p; Ai = A.i
    parent = S.parent
    S.pinv = pinv = ialloc(m + n) # allocate pinv,
    S.leftmost = leftmost = ialloc(m) # and leftmost
    w = ialloc(m + 3 * n) # get workspace
    next = w
    head = w
    head_offset = m
    tail = w
    tail_offset = m + n
    nque = w
    nque_offset = m + 2 * n
    for k in range(n):
        head[head_offset + k] = -1 # queue k is empty
    for k in range(n):
        tail[tail_offset + k] = -1
    for k in range(n):
        nque[nque_offset + k] = 0
    for i in range(m):
        leftmost[i] = -1
    k = n - 1
    while k >= 0:
        for p in range(Ap[k], Ap[k + 1]):
            leftmost[Ai[p]] = k # leftmost[i] = min(find(A(i,:)))
        k-=1
    i = m - 1
    while i >= 0: # scan rows in reverse order
        pinv[i] = -1 # row i is not yet ordered
        k = leftmost[i]
        if k == -1:
            i-=1
            continue # row i is empty
        if nque[nque_offset + k] == 0:
            tail[tail_offset + k] = i # first row in queue k
        nque[nque_offset + k]+=1
        next[i] = head[head_offset + k] # put i at head of queue k
        head[head_offset + k] = i
        i-=1
    S.lnz = 0
    S.m2 = m
    for k in range(n): # find row permutation and nnz(V)
        i = head[head_offset + k] # remove row i from queue k
        S.lnz+=1 # count V(k,k) as nonzero
        if i < 0:
            i = S.m2 # add a fictitious row
            S.m2+=1
        pinv[i] = k # associate row i with V(:,k)
        nque[nque_offset + k]-=1
        if nque[nque_offset + k] <= 0:
            continue # skip if V(k+1:m,k) is empty
        S.lnz += nque[nque_offset + k] # nque [nque_offset+k] is nnz (V(k+1:m,k))
        pa = parent[k]
        if pa != -1: # move all rows to parent of k
            if nque[nque_offset + pa] == 0:
                tail[tail_offset + pa] = tail[tail_offset + k]
            next[tail[tail_offset + k]] = head[head_offset + pa]
            head[head_offset + pa] = next[i]
            nque[nque_offset + pa] += nque[nque_offset + k]
    for i in range(m):
        if pinv[i] < 0:
            pinv[i] = k
            k+=1
    w = None
    return True


def cs_sqr(order, A, qr):
    """Symbolic QR or LU ordering and analysis.

    @param order: ordering method to use (0 to 3)
    @param A: column-compressed matrix
    @param qr: analyze for QR if true or LU if false
    @return: symbolic analysis for QR or LU, null on error
    """
    ok = True
    if not CS_CSC(A):
        return None # check inputs
    n = A.n
    S = css() # allocate result S
    S.q = cs_amd(order, A) # fill-reducing ordering
    if order > 0 and S.q == None:
        return None
    if qr: # QR symbolic analysis
        C = cs_permute(A, None, S.q, False) if order > 0 else A
        S.parent = cs_etree(C, True) # etree of C'*C, where C=A(:,q)
        post = cs_post(S.parent, n)
        S.cp = cs_counts(C, S.parent, post, True) # col counts chol(C'*C)
        ok = C != None and S.parent != None and S.cp != None and _cs_vcount(C, S)
        if ok:
            S.unz = 0
            for k in range(n):
                S.unz += S.cp[k]
        ok = ok and S.lnz >= 0 and S.unz >= 0 # int overflow guard
    else:
        S.unz = 4 * A.p[n] + n # for LU factorization only,
        S.lnz = S.unz # guess nnz(L) and nnz(U)
    return S if ok else None # return result S


def cs_symperm(A, pinv, values):
    """Permutes a symmetric sparse matrix. C = PAP' where A and C are symmetric.

    @param A: column-compressed matrix (only upper triangular part is used)
    @param pinv: size n, inverse permutation
    @param values: allocate pattern only if false, values and pattern otherwise
    @return: C = PAP', null on error
    """
    if not CS_CSC(A):
        return None # check inputs
    n, Ap, Ai, Ax = A.n, A.p, A.i, A.x
    C = cs_spalloc(n, n, Ap[n], values and (Ax != None), False) # alloc result
    w = ialloc(n) # get workspace
    Cp, Ci, Cx = C.p, C.i, C.x
    for j in range(n): # count entries in each column of C
        j2 = pinv[j] if pinv != None else j # column j of A is column j2 of C
        for p in range(Ap[j], Ap[j + 1]):
            i = Ai[p]
            if i > j:
                continue # skip lower triangular part of A
            i2 = pinv[i] if pinv != None else i # row i of A is row i2 of C
            w[max(i2, j2)]+=1 # column count of C
    cs_cumsum(Cp, w, n) # compute column pointers of C
    for j in range(n):
        j2 = pinv[j] if pinv != None else j # column j of A is column j2 of C
        for p in range(Ap[j], Ap[j + 1]):
            i = Ai[p]
            if i > j:
                continue # skip lower triangular part of A
            i2 = pinv[i] if pinv != None else i # row i of A is row i2 of C
            q = w[max(i2, j2)]
            w[max(i2, j2)]+=1
            Ci[q] = min(i2, j2);
            if Cx != None:
                Cx[q] = Ax[p]
    return C


def cs_tdfs(j, k, head, head_offset, next, next_offset, post, post_offset,
            stack, stack_offset):
    """Depth-first search and postorder of a tree rooted at node j

    @param j: postorder of a tree rooted at node j
    @param k: number of nodes ordered so far
    @param head: head[i] is first child of node i; -1 on output
    @param head_offset: the index of the first element in array head
    @param next: next[i] is next sibling of i or -1 if none
    @param next_offset: the index of the first element in array next
    @param post: postordering
    @param post_offset: the index of the first element in array post
    @param stack: size n, work array
    @param stack_offset: the index of the first element in array stack
    @return new value of k, -1 on error
    """
    top = 0
    if head == None or next == None or post == None or stack == None:
        return -1 # check inputs
    stack[stack_offset + 0] = j # place j on the stack
    while top >= 0: # while (stack is not empty)
        p = stack[stack_offset + top] # p = top of stack
        i = head[head_offset + p] # i = youngest child of p
        if i == -1:
            top-=1 # p has no unordered children left
            post[post_offset + k] = p # node p is the kth postordered node
            k+=1
        else:
            head[head_offset + p] = next[next_offset + i] # remove i from children of p
            top+=1
            stack[stack_offset + top] = i # start dfs on child node i
    return k


def cs_transpose(A, values):
    """Computes the transpose of a sparse matrix, C =A';

    @param A: column-compressed matrix
    @param values: pattern only if false, both pattern and values otherwise
    @return: C=A', null on error
    """
    if not CS_CSC(A):
        return None # check inputs
    m, n, Ap, Ai, Ax = A.m, A.n, A.p, A.i, A.x
    C = cs_spalloc(n, m, Ap[n], values and (Ax != None), False) # allocate result
    w = ialloc(m) # get workspace
    Cp, Ci, Cx = C.p, C.i, C.x
    for p in range(Ap[n]):
        w[Ai[p]]+=1 # row counts
    cs_cumsum(Cp, w, m) # row pointers
    for j in range(n):
        for p in range(Ap[j], Ap[j + 1]):
            q = w[Ai[p]]
            w[Ai[p]]+=1
            Ci[q] = j # place A(i,j) as entry C(j,i)
            if Cx != None:
                Cx[q] = Ax[p]
    return C


def cs_updown(L, sigma, C, parent):
    """Sparse Cholesky rank-1 update/downdate, L*L' + sigma*w*w' (sigma = +1 or -1)

    @param L: factorization to update/downdate
    @param sigma: +1 for update, -1 for downdate
    @param C: the vector c
    @param parent: the elimination tree of L
    @return: true if successful, false on error
    """
    beta = beta2 = 1
    if not CS_CSC(L) or not CS_CSC(C) or parent == None:
        return False # check inputs
    Lp, Li, Lx, n = L.p, L.i, L.x, L.n
    Cp, Ci, Cx = C.p, C.i, C.x
    p = Cp[0]
    if p >= Cp[1]:
        return True # return if C empty
    w = xalloc(n) # get workspace
    f = Ci[p]
    while p < Cp[1]:
        f = min(f, Ci[p]) # f = min (find (C))
        p+=1
    j = f
    while j != -1:
        w[j] = 0 # clear workspace w
        j = parent[j]
    for p in range(Cp[0], Cp[1]):
        w[Ci[p]] = Cx[p] # w = C
    j = f
    while j != -1: # walk path f up to root
        p = Lp[j]
        alpha = w[j] / Lx[p] # alpha = w(j) / L(j,j)
        beta2 = beta * beta + sigma * alpha * alpha
        if beta2 <= 0:
            break # not positive definite
        beta2 = sqrt(beta2)
        delta = (beta / beta2) if sigma > 0 else (beta2 / beta)
        gamma = sigma * alpha / (beta2 * beta)
        Lx[p] = delta * Lx[p] + ((gamma * w[j]) if sigma > 0 else 0)
        beta = beta2
        p+=1
        while p < Lp[j + 1]:
            w1 = w[Li[p]]
            w[Li[p]] = w2 = w1 - alpha * Lx[p]
            Lx[p] = delta * Lx[p] + gamma * (w1 if sigma > 0 else w2)
            p+=1
        j = parent[j]
    return beta2 > 0


def cs_usolve(U, x):
    """Solves an upper triangular system Ux=b, where x and b are dense vectors.
    The diagonal of U must be the last entry of each column.

    @param U: upper triangular matrix in column-compressed form
    @param x: size n, right hand side on input, solution on output
    @return: true if successful, false on error
    """
    if not CS_CSC(U) or x == None:
        return False # check inputs
    n, Up, Ui, Ux = U.n, U.p, U.i, U.x
    j = n - 1
    while j >= 0:
        x[j] /= Ux[Up[j + 1] - 1];
        for p in range(Up[j], Up[j + 1] - 1):
            x[Ui[p]] -= Ux[p] * x[j]
        j-=1
    return True


def cs_spalloc(m, n, nzmax, values, triplet):
    """Allocate a sparse matrix (triplet form or compressed-column form).

    @param m: number of rows
    @param n: number of columns
    @param nzmax: maximum number of entries
    @param values: allocate pattern only if false, values and pattern otherwise
    @param triplet: compressed-column if false, triplet form otherwise
    @return: sparse matrix
    """
    A = cs() # allocate the cs object
    A.m = m # define dimensions and nzmax
    A.n = n
    A.nzmax = nzmax = max(nzmax, 1)
    A.nz = 0 if triplet else -1 # allocate triplet or comp.col
    A.p = ialloc(nzmax) if triplet else ialloc(n + 1)
    A.i = ialloc(nzmax)
    A.x = xalloc(nzmax) if values else None
    return A


def _copy(src, dest, length):
    for i in range(length):
        dest[i] = src[i]


def cs_sprealloc(A, nzmax):
    """Change the max # of entries a sparse matrix can hold.

    @param A: column-compressed matrix
    @param nzmax: new maximum number of entries
    @return: true if successful, false on error
    """
    if A == None:
        return False
    if nzmax <= 0:
        nzmax = A.p[A.n] if CS_CSC(A) else A.nz
    Ainew = ialloc(nzmax)
    length = min(nzmax, len(A.i))
    _copy(A.i, Ainew, length)
    A.i = Ainew
    if CS_TRIPLET(A):
        Apnew = ialloc(nzmax)
        length = min(nzmax, len(A.p))
        _copy(A.p, Apnew, length)
        A.p = Apnew
    if A.x != None:
        Axnew = xalloc(nzmax)
        length = min(nzmax, len(A.x))
        _copy(A.x, Axnew, length)
        A.x = Axnew
    A.nzmax = nzmax
    return True


def cs_dalloc(m, n):
    """Allocate a Dcsd object (a Dulmage-Mendelsohn decomposition).

    @param m: number of rows of the matrix A to be analyzed
    @param n: number of columns of the matrix A to be analyzed
    @return: Dulmage-Mendelsohn decomposition
    """
    D = csd()
    D.p = ialloc(m)
    D.r = ialloc(m + 6)
    D.q = ialloc(n)
    D.s = ialloc(n + 6)
    D.cc = ialloc(5)
    D.rr = ialloc(5)
    return D


def cs_utsolve(U, x):
    """Solves a lower triangular system U'x=b, where x and b are dense vectors.
    The diagonal of U must be the last entry of each column.

    @param U: upper triangular matrix in column-compressed form
    @param x: size n, right hand side on input, solution on output
    @return: true if successful, false on error
    """
    if not CS_CSC(U) or x == None:
        return False # check inputs
    n, Up, Ui, Ux = U.n, U.p, U.i, U.x
    for j in range(n):
        for p in range(Up[j], Up[j + 1] - 1):
            x[j] -= Ux[p] * x[Ui[p]]
        x[j] /= Ux[Up[j + 1] - 1]
    return True


def ialloc(n):
    return [0]*n


def xalloc(n):
    return [0.0]*n
