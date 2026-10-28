import numpy as np

# AUXILIARES

def elim_gaussiana(A):
    m = A.shape[0]
    n = A.shape[1]
    Ac = A.copy()

    for j in range(m - 1):
        for i in range(j + 1, m):
            Ac[i, j] = -Ac[i, j] / Ac[j, j]

            for k in range(j + 1, n):
                Ac[i, k] = Ac[i, k] + Ac[i, j] * Ac[j, k]

    return Ac

def norma(x, p):
   # x = np.array(x, dtype=float)  # convierte cualquier entrada a array
    if p == 'inf':
        actual = 0
        for i in x:
            if abs(i) > actual:
                actual = abs(i)
        return actual
    else:
        res: float = 0
        for i in x:
            res = res + abs(i)**p
        res = res**(1/p)
        return res

def normaExacta(A, p=[1,'inf']):
    maximo = 0
    if p == 1:
        return normaExacta(traspuesta(A), 'inf')
    elif p == 'inf':
        for x in A:
            actual = norma(x,1)
            if maximo < actual:
                maximo = actual
        return maximo
    elif p == [1,'inf']:
        return [normaExacta(A,1), normaExacta(A,'inf')]

def productoInterno(u,v,tol=1e-12):
    res = 0
    n = u.shape[0]
    for fil in range(n):
        res += u[fil]*v[fil]
    return res

def multiplicacionMatricial(A, B, tol=1e-12):
    # Guardamos las dimensiones originales
    A_vector = A.ndim == 1
    B_vector = B.ndim == 1

    if A_vector:
        A = A.reshape(1, -1)
    if B_vector:
        B = B.reshape(-1, 1)

    filas_A = A.shape[0]
    filas_B = B.shape[0]
    columnas_A = A.shape[1]
    columnas_B = B.shape[1]

    if columnas_A != filas_B:
        return None

    res = []
    for fil in range(filas_A):
        vector = []
        for col in range(columnas_B):
            vector.append(
                productoInterno(A[fil, :], B[:, col])
            )
        res.append(vector)

    res = np.array(res)

    # Devolvemos el resultado con las dimensiones esperadas
    if A_vector and B_vector:
        return res[0, 0]

    if A_vector:
        return res[0, :]

    if B_vector:
        return res[:, 0]

    return res


def transformacion_f(v,A):
    n = v.shape

    if 0 >= norma(v,2):
        v = np.zeros(n)
        return v
    else: 
        w = multiplicacionMatricial(A, v)
        w /= norma(w,2)
        return w

def productoExterno(u,v):
    res = []
    for i in range(len(u)):
        vector = []
        for j in range(len(v)):
            vector.append(u[i]*v[j])
        vector = np.array(vector)
        res.append(vector)
    res = np.array(res)
    return res

def signo(a):
    if a < 0:
        return -1
    else:
        return 1

def matriz_householder(x, tol=1e-12):
    if x.ndim != 1:
        return None

    if norma(x,2) < tol:
        return None
    
    e = np.zeros(len(x))
    e[0] = 1
    s = 1 if x[0] >= 0 else -1
    v = x + s * norma(x, 2) * e

    if norma(v,2) < tol:
        return None
    
    v = v / norma(v,2)
    H = np.eye(len(x)) - 2*productoExterno(v,v)

    return H

def traspuesta(A):   
    if A is None:
        return None
    filas = A.shape[0]
    columnas = A.shape[1]
    fil = 0
    col = 0
    res: list = []
    for fil in range(0, columnas):
        vector = []
        for col in range(0, filas):
            vector.append(A[col][fil])
        vector = np.array(vector)
        res.append(vector)
    res = np.array(res)
    return res

## FUNCIONES L06
        
def metpot2k(A, tol=1e-15, K=1000):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    n = A.shape[1]
    v = np.random.randn(n)
    v = v / norma(v, 2)

    w = transformacion_f(v, A)
    k = 1

    while norma(w - v, 2) > tol and norma(w + v, 2) > tol and k < K:
        v = w
        w = transformacion_f(v, A)
        k += 1

    lbd = productoInterno(w, multiplicacionMatricial(A, w))

    return w, lbd, k



def diagRH(A, tol=1e-15,K=1000):
    v, lbd, _ = metpot2k(A, tol, K)
    #v = v.flatten()

    H = matriz_householder(v)
    n = A.shape[0]
    Am = np.zeros((n,n))
    if n==2:
        S = H
        D = multiplicacionMatricial(multiplicacionMatricial(H, A), traspuesta(H))
    else:
        B = multiplicacionMatricial(multiplicacionMatricial(H, A), traspuesta(H))
        Am = B[1:n,1:n]
        Sm, Dm = diagRH(Am, tol, K)
        D = np.zeros((n,n))
        D[0,0] = lbd
        D[1:n,1:n] = Dm

        S = np.zeros((n,n))
        S[0,0] = 1
        S[1:n,1:n] = Sm
        S = multiplicacionMatricial(H, S)
    return S, D

# Tests L08
def svd_reducida(A, k="max", tol=1e-15):
    m = A.shape[0]
    n = A.shape[1]

    # Caso 1: m >= n. Calculamos primero V
    if m >= n:
        AtA = multiplicacionMatricial(traspuesta(A), A)
        hatV, E = diagRH(AtA, tol=tol)

        # Extraemos los autovalores de la diagonal
        lbd = np.zeros(n)
        for i in range(n):
            lbd[i] = E[i, i]

        # Ordenamos los autovalores de mayor a menor,
        # intercambiando también los autovectores asociados
        for i in range(n):
            for j in range(i + 1, n):
                if lbd[j] > lbd[i]:
                    aux = lbd[i]
                    lbd[i] = lbd[j]
                    lbd[j] = aux

                    aux = hatV[:, i].copy()
                    hatV[:, i] = hatV[:, j]
                    hatV[:, j] = aux

        # Contamos los autovalores que no son nulos
        r = 0
        for i in range(n):
            if lbd[i] > tol:
                r += 1

        # Si se especifica k, retenemos como máximo k valores
        if k != "max":
            r = min(r, k)

        hatSig = np.zeros(r)
        V = np.zeros((n, r))

        for j in range(r):
            hatSig[j] = np.sqrt(lbd[j])
            V[:, j] = hatV[:, j]

        # Calculamos U a partir de A V = U Sigma
        B = multiplicacionMatricial(A, V)
        U = np.zeros((m, r))

        for j in range(r):
            U[:, j] = B[:, j] / hatSig[j]

    # Caso 2: m < n. Calculamos primero U
    else:
        AAt = multiplicacionMatricial(A, traspuesta(A))
        hatU, E = diagRH(AAt, tol=tol)

        # Extraemos los autovalores de la diagonal
        lbd = np.zeros(m)
        for i in range(m):
            lbd[i] = E[i, i]

        # Ordenamos los autovalores de mayor a menor,
        # intercambiando también los autovectores asociados
        for i in range(m):
            for j in range(i + 1, m):
                if lbd[j] > lbd[i]:
                    aux = lbd[i]
                    lbd[i] = lbd[j]
                    lbd[j] = aux

                    aux = hatU[:, i].copy()
                    hatU[:, i] = hatU[:, j]
                    hatU[:, j] = aux

        # Contamos los autovalores que no son nulos
        r = 0
        for i in range(m):
            if lbd[i] > tol:
                r += 1

        # Si se especifica k, retenemos como máximo k valores
        if k != "max":
            r = min(r, k)

        hatSig = np.zeros(r)
        U = np.zeros((m, r))

        for j in range(r):
            hatSig[j] = np.sqrt(lbd[j])
            U[:, j] = hatU[:, j]

        # Calculamos V a partir de A^T U = V Sigma
        B = multiplicacionMatricial(traspuesta(A), U)
        V = np.zeros((n, r))

        for j in range(r):
            V[:, j] = B[:, j] / hatSig[j]

    return U, hatSig, V

#labo 5
def QR_con_HH(A,tol=1e-12,extras=False):
    m = A.shape[0]
    n = A.shape[1]
    R = A.copy()
    Q = np.eye(m)

    if n > m:
        return None

    # Construyo la primera matriz HH
    for i in range(n):
        x = R[i:,i]
        Hp = matriz_householder(x, tol)
        H = np.eye(n)
        H[i:,i:] = Hp
        R = multiplicacionMatricial(H, R)
        if i == 0:
            Q = H
        else:
            Q = multiplicacionMatricial(Q,H)
        if R is None:
            return Q, R
    return Q, R

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    m = A.shape[0]
    n = A.shape[1]
    cant_ops = 0

    Rm = np.eye(n)
    Qm = np.zeros((m, n))

    # construyo las matrices moño
    Qm[:, 0] = A[:, 0]

    norma = np.sqrt(productoInterno(Qm[:, 0], Qm[:, 0]))

    if norma < tol:
        return None

    for j in range(1, n):  # itero por cada columna de A
        Qm[:, j] = A[:, j]

        for k in range(0, j):  # itero por cada columna ya construida
            Rm[k, j] = productoInterno(Qm[:, k], Qm[:, j]) / productoInterno(Qm[:, k], Qm[:, k])
            Qm[:, j] = Qm[:, j] - Rm[k, j] * Qm[:, k]

        # uso la tolerancia para detectar una columna nula
        norma = np.sqrt(productoInterno(Qm[:, j], Qm[:, j]))

        if norma < tol:
            return None

    # creo matrices normalizadoras
    Qn = np.zeros((n, n))
    Rn = np.zeros((n, n))

    for i in range(0, n):
        Qn[i, i] = 1 / np.sqrt(productoInterno(Qm[:, i], Qm[:, i]))
        Rn[i, i] = np.sqrt(productoInterno(Qm[:, i], Qm[:, i]))

    Q = multiplicacionMatricial(Qm, Qn)
    R = multiplicacionMatricial(Rn, Rm)

    if retorna_nops:
        return Q, R, cant_ops

    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo=='GS':
        Q, R = QR_con_GS(A, tol)
    elif metodo=='RH':
        Q, R = QR_con_HH(A, tol)
    else:
        return None
    return Q, R

#aux
def rq(M):
    #Recibe una matriz de 3x3 y devuelve las matrices K y R de la descomposición RQ.
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    M_mod = (P, multiplicacionMatricial(traspuesta(M), P))

    Q_qr, R_qr = QR_con_HH(M_mod)

    K = multiplicacionMatricial(P, multiplicacionMatricial(traspuesta(R_qr), P))
    R = multiplicacionMatricial(P, multiplicacionMatricial(traspuesta(Q_qr), P))

    return K, R