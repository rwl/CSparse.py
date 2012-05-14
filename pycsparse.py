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

from math import sqrt


CS_VER = 1 # PyCSparse Version 1.0.0
CS_SUBVER = 0
CS_SUBSUB = 0
CS_DATE = "May 14, 2012" # PyCSparse release date
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
    cs_fkeep(C, cs_diag(), None) # drop diagonal entries
    Cp = C.p
    cnz = Cp[n]
    P = [0]*(n + 1) # allocate result
    W = [0]*(8 * (n + 1)) # get workspace
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
        k1 = 1
        while k1 <= elenk + 1:
            if k1 > elenk:
                e = k # search the nodes in k
                pj = p # list of nodes starts at Ci[pj]
                ln = len[k] - elenk # length of list of nodes in k
            else:
                e = Ci[p] # search the nodes in e
                p+=1
                pj = Cp[e]
                ln = len[e] # length of list of nodes in e
            k2 = 1
            while k2 <= ln:
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
                k2+=1
            if e != k:
                Cp[e] = CS_FLIP(k) # absorb e into k
                w[w_offset + e] = 0 # e is now a dead element
            k1+=1
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
        for pk in range(pk1, pk < pk2):
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
            continue # skip if j is an element
        next[next_offset + j] = head[head_offset + Cp[j]] # place j in list of its parent
        head[head_offset + Cp[j]] = j
        j-=1
    e = n
    while e >= 0: # place elements in lists
        if nv[nv_offset + e] <= 0:
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
    c = [0]*(2 * n) # get int workspace
    x = [0.0]*n # get double workspace
    cp = S.cp
    pinv = S.pinv
    parent = S.parent
    C = cs_symperm(A, pinv, True) if pinv != null else A
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
    x = [0.0]*n # get workspace
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
    w = [0]*n # get workspace
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
    m = AT.n, n = AT.m, ATp = AT.p, ATi = AT.i
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
    delta = colcount = [0]*n # allocate result
    w = [0]*s # get workspace
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
    cs_bfs(A, n, wi, wj, q, imatch, imatch_offset, jmatch, 0, 1) # find C1, R1 from C0
    ok = cs_bfs(A, m, wj, wi, p, jmatch, 0, imatch, imatch_offset, 3) # find R3, C3 from R0
    if not ok:
        return None
    cs_unmatched(n, wj, q, cc, 0) # unmatched set C0
    cs_matched(n, wj, imatch, imatch_offset, p, q, cc, rr, 1, 1) # set R1 and C1
    cs_matched(n, wj, imatch, imatch_offset, p, q, cc, rr, 2, -1) # set R2 and C2
    cs_matched(n, wj, imatch, imatch_offset, p, q, cc, rr, 3, 3) # set R3 and C3
    cs_unmatched(m, wi, p, rr, 3) # unmatched set R0
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
