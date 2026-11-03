# -*- coding: utf-8 -*-
"""
Grupo: A=TR

Turno: Mañana

Integrantes: 

"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import random

#%% Labo 00

def esCuadrada(A):
    res: bool = True
    filas = A.shape[0]
    columnas = A.shape[1]
    if not (filas == columnas):
        res = False
    return res


def triangSup(A):
    filas = A.shape[0]
    columnas = A.shape[1]
    res: list = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            if i < j:
                fila.append(A[i, j])
            else:
                fila.append(0)
        res.append(fila)
    res = np.array(res)
    return res


def triangInf(A):
    filas = A.shape[0]
    columnas = A.shape[1]
    res: list = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            if j < i:
                fila.append(A[i, j])
            else:
                fila.append(0)
        res.append(fila)
    res = np.array(res)
    return res

def diagonal(A):
    filas = A.shape[0]
    columnas = A.shape[1]
    maximo = min([filas,columnas])
    res: list = []
    for indice in range(0,maximo):
        vector = []
        vector.append(A[indice][indice])
        res.append(vector)
    res = np.array(res)
    return res


def traza(A):
    filas = A.shape[0]
    columnas = A.shape[1]
    maximo = min([filas, columnas])
    res: int = 0
    for indice in range(0, maximo):
        res += A[indice, indice]
    return res


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

def esSimetrica(A):
    res: bool = True
    filas = A.shape[0]
    columnas = A.shape[1]
    trasp = traspuesta(A)
    for fil in range(filas):
        for col in range(columnas):
            if not (A[fil, col] == trasp[fil, col]):
                res = False
    return res


def calcularAx(A, x):
    columnas_A = A.shape[1]
    filas_x = x.shape[0]
    res: list = []
    if filas_x == columnas_A:
        for i in range(filas_x):
            suma = 0
            for j in range(filas_x):
                prod = A[i, j] * x[j]
                suma += prod
            res.append(suma)
    res = np.array(res)
    return res


def intercambiarFilas(A, i, j):
    columnas = A.shape[1]
    for col in range(columnas):
        valor = A[i, col]
        A[i, col] = A[j, col]
        A[j, col] = valor
    return A


def sumar_fila_multiplo(A, i, j, s):
    columnas = A.shape[1]
    for col in range(columnas):
        A[i, col] += A[j, col]*s
    return A


def esDiagonalmenteDominante(A):
    res: bool = True
    longitud = A.shape[0]
    for i in range(longitud):
        suma = 0
        for j in range(longitud):
            if i != j:
                suma += abs(A[i, j])
        print(suma)
        print(abs(A[i, i]))
        if abs(A[i, i]) < suma:
            res = False
    return res


def matrizCirculante(v):
    res: list = []
    longitud = v.shape[0]
    res.append(v)
    vector = v
    for ind in range(1, longitud):
        vector = desplazarDerecha(vector)
        res.append(vector)
    res = np.array(res)
    return res


def desplazarDerecha(v):
    res: list = []
    ultimo: int = v[v.shape[0] - 1]
    v = v[:v.shape[0]-1]
    res.append(ultimo)
    for ind in range(0, v.shape[0]):
        res.append(v[ind])
    res = np.array(res)
    return res


def matrizVandermonde(v):
    res: list = []
    n: int = v.shape[0]
    for i in range(0, n):
        vector = v**i
        res.append(vector)
    res = np.array(res)
    return res


def numeroAureo(n):
    f0: int = 0
    f1: int = 1
    a = f0
    b = f1
    for i in range(n):
        temp = a + b
        a = b
        b = temp
    return b/a


def matfibo(m):
    F = [0, 1]
    for k in range(2, m+1):
        F.append(F[k-1] + F[k-2])
    return F


def hilbert(n):
    res = []
    for i in range(n):
        fila = []
        for j in range(n):
            fila.append(1/(i+j+1))
        res.append(fila)
    return res

def evaluar_polinomio1(x):
    return x*5 - x*4 + x*3 - x*2 + x - 1

""" LO COMENTO PARA QUE NO ESTORBE DURANTE LA EJECUCIÓN
x = np.linspace(-1, 1, 100)  # GENERA 100 PUNTOS ENTRE -1 , 1


y1 = evaluar_polinomio1(x)  # EVALUA EL POL


plt.plot(x, y1, label='x^5 - x^4 + x^3 - x^2 + x - 1')
plt.xlabel('x')  # muestra el nombre del ejex
plt.ylabel('Valor del polinomio')  # nombre eje y
plt.title('Polinomio P1(x) entre -1 y 1')  # le pone el titulo
plt.legend()  # el cosito q te dice q la funcion azul por ej es esta
plt.grid(True)  # cuadricula del fondo
plt.show()
"""

def row_echelon(M):
    """
    Retorna la matriz escalonada por filas
    utilizando pivoteo parcial.
    """
    A = np.array(M, dtype=np.float32)

    f, c = A.shape

    # Si A no tiene filas o columnas, ya está escalonada
    if f == 0 or c == 0:
        return A

    # Buscamos el elemento de mayor módulo en la primera columna
    i = 0
    for k in range(1, f):
        if abs(A[k, 0]) > abs(A[i, 0]):
            i = k

    # Si toda la primera columna es cero,
    # escalonamos recursivamente desde la segunda columna
    if A[i, 0] == 0:
        B = row_echelon(A[:, 1:])

        res = np.zeros((f, c))
        res[:, 0] = A[:, 0]

        if c > 1:
            res[:, 1:] = B

        return res

    # Intercambiamos las filas para colocar el mejor pivote
    if i > 0:
        aux = A[0, :].copy()
        A[0, :] = A[i, :]
        A[i, :] = aux

    # Eliminamos los elementos debajo del pivote
    for k in range(1, f):
        factor = A[k, 0] / A[0, 0]
        A[k, :] = A[k, :] - factor * A[0, :]

    # Escalonamos recursivamente la submatriz restante
    B = row_echelon(A[1:, 1:])

    # Reconstruimos el resultado sin np.block
    res = np.zeros((f, c))
    res[0, :] = A[0, :]

    for k in range(1, f):
        res[k, 0] = A[k, 0]

    if f > 1 and c > 1:
        res[1:, 1:] = B

    return res

#%% Funciones Auxiliares

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

def productoInterno(u,v,tol=1e-12):
    res = 0
    n = u.shape[0]
    for fil in range(n):
        res += u[fil]*v[fil]
    return res

def calcularxA(x, A):
    columnas = A.shape[1]
    res: list = []
    for col in range(columnas):
        res.append(productoInterno(x, A[col]))
    res = np.array(res)
    return res


def multiplicacionMatricial(A, B, tol=1e-12):
    columnas_A = A.shape[1]
    filas_A = A.shape[0]
    filas_B = B.shape[0]
    columnas_B = B.shape[1]
    if columnas_A != filas_B:
        return None
    else:
        res = []
        for fil in range(filas_A):
            vector = []
            for col in range(columnas_B):
                vector.append(productoInterno(A[fil, :], B[:, col]))
            res.append(vector)
        res = np.array(res)
        return res

def signo(a):
    if a < 0:
        return -1
    else:
        return 1

def matriz_householder(x, tol=1e-12):
    if len(x.shape) != 1:
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

def transformacion_f(v,A):
    n = v.shape

    if 0 >= norma(v,2):
        v = np.zeros(n)
        return v
    else: 
        w = calcularAx(A, v)
        w /= norma(w,2)
        return w
    
#%% Labo 01

def error(x, y):
    res: int
    x64 = float(x)
    y64 = float(y)
    res = x64 - y64
    if res < 0:
        res = -res  
    return res

def error_relativo(x,y):
    res: int
    x64 = float(x)
    err_abs: int = error(x,y)
    res = err_abs / abs(x64) if x64 != 0 else float('nan')
    return res

def matricesIguales(A, B):
    res: bool = True
    for fil in range(A.shape[0]):
        for col in range(A.shape[1]):
            if not(error(A[fil][col],B[fil][col]) <= 1e-07):
                res = False
    return res

#%% Labo 02

def rota(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def escala(s):
    res: list = []
    for i in range(len(s)):
        vector = []
        for j in range(len(s)):
            if i == j:
                vector.append(s[i])
            else:
                vector.append(0)
        vector = np.array(vector)
        res.append(vector)
    res = np.array(res)
    return res

def rota_y_escala(theta, s):
    return multiplicacionMatricial(escala(s), rota(theta))

def afin(theta, s, b):
    res: list = []
    res.append(np.array([rota_y_escala(theta,s)[0][0], rota_y_escala(theta,s)[0][1], b[0]]))
    res.append(np.array([rota_y_escala(theta,s)[1][0], rota_y_escala(theta,s)[1][1], b[1]]))
    res.append(np.array([0,0,1]))
    return np.array(res)

def trans_afin(v, theta, s, b):
    xy1 = np.array([[v[0]],[v[1]],[1]])
    res = multiplicacionMatricial(afin(theta, s, b), xy1)
    return res[:2, 0]

#%% Labo 03

def norma(x, p):
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


def normaliza(X, p):    
    res = []
    for v in X:
        v = np.array(v, dtype=float)
        n = norma(v, p)
        if n != 0:
            res.append(v / n)
        else:
            res.append(v)
    return res    

def normaMatMC(A, q, p, Np):
    maximoValor: float = 0
    maximoVector = None
    n = A.shape[1]
    for it in range(Np):
        x = np.random.randn(n)
        x = x / norma(x, p)
        Ax = calcularAx(A, x)
        valorActual = norma(Ax, q) 
        if maximoValor < valorActual:
            maximoValor = valorActual
            maximoVector = x
    return maximoValor, maximoVector

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

def condMC(A, p, Np):
    return normaMatMC(A, p, p, Np)[0]*normaMatMC(inversa(A), p, p, Np)[0]


def condExacta(A, p):
    return normaExacta(A, p)*normaExacta(inversa(A), p)

#%% Labo 04

def calculaLU(A):
    if A is None:
        return None, None, 0
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        return None, None, 0
    
    L = np.eye(m,n)

    for j in range(m-1): 

        Lm = np.eye(m,n)

        if Ac[j,j] == 0:
            return None, None, 0
        
        for i in range(j + 1, m): 
            cant_op += 1

            multiplicador = Ac[i,j] / Ac[j,j]

            L[i,j] = multiplicador

            Ac[i,j:] = Ac[i,j:] - multiplicador * Ac[j,j:]

        
        cant_op += 2 * (m-j-1)**2

    if Ac[m-1,m-1] == 0:
        return None, None, 0

    U = Ac
    return L, U, cant_op

def res_tri(L, b, inferior=True):
    m = L.shape[0]
    x = np.zeros(m)

    if inferior:
        for i in range(m):
            suma = 0
            for j in range(i):
                suma += L[i,j] * x[j]
            x[i] = (b[i] - suma) / L[i,i]

    else:
        for i in range(m-1, -1, -1):
            suma = 0
            for j in range(i+1, m):
                suma += L[i,j] * x[j]
            x[i] = (b[i] - suma) / L[i,i]

    return x

def inversa(A):

    n = A.shape[0]

    aumentada = np.zeros((n, 2*n))
    aumentada[:, :n] = np.array(A)
    aumentada[:, n:] = np.eye(n)

    for i in range(n):

        j = i + 1

        while aumentada[i,i] == 0 and j < n:

            if aumentada[j,i] != 0:
                aumentada[[i,j]] = aumentada[[j,i]]

            j += 1

        if aumentada[i,i] == 0:
            return None

        pivote = aumentada[i,i]

        aumentada[i] = aumentada[i] / pivote

        for j in range(n):

            if j != i:

                factor = aumentada[j,i]

                aumentada[j] = aumentada[j] - factor * aumentada[i]

    return aumentada[:,n:]

def calculaLDV(A):
    L, U, V = calculaLU(A)[0], calculaLU(A)[1], traspuesta(calculaLU(A)[0])

    if U is None:
        return L, None, None

    m = A.shape[0]
    D = np.zeros((m,m))
    for i in range(m):
        D[i,i] = U[i,i]
    return L, D, V

def esSDP(A, atol=1e-8):
    L, D, _ = calculaLDV(A)
    if L is None:
        return False
    n = A.shape[0]
    res = True
    for i in range(n):
        if not D[i, i] > atol:
            return False
    return res

def calculaCholesky(A,atol=1e-10):
    if not esSDP(A):
        return None
    L, D, V = calculaLDV(A)
    R = multiplicacionMatricial(L, np.sqrt(D))
    return R

#%% Labo 05

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

def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo=='GS':
        Q, R = QR_con_GS(A, tol)
    elif metodo=='RH':
        Q, R = QR_con_HH(A, tol)
    else:
        return None
    return Q, R

#%% Labo 06

def metpot2k(A, tol=1e-15, K=1000):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
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

    lbd = productoInterno(w, calcularAx(A, w))

    return w, lbd, k

def diagRH(A, tol=1e-15,K=1000):
    v, lbd, _ = metpot2k(A, tol, K)
    v = v.reshape(-1)

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

#%% Labo 07

def transiciones_al_azar_continuas(n):
    A = np.zeros((n,n))

    for j in range(n):
        for i in range(n):
            A[i,j] = random.uniform(0,1)

    for j in range(n):
        suma = 0
        for i in range(n):
            suma = suma + A[i, j]

        for i in range(n):
            A[i, j] = A[i, j] / suma

    return A 

def transiciones_al_azar_uniformes(n,thres):
    A = np.zeros((n,n))

    for j in range(n):
        for i in range(n):
            A[i,j] = random.uniform(0,1)
            if thres > A[i,j]:
                A[i,j] = 1
            else:
                A[i,j] = 0

    for j in range(n):
        suma = 0
        for i in range(n):
            suma = suma + A[i, j]

        for i in range(n):
            if suma != 0:
                A[i, j] = A[i, j] / suma
            else:
                A[i, j] = 1/n # Distribución uniforme

    return A 

def nucleo(A,tol=1e-15):
    # Hago la diagonalizacion
    C, D = diagRH(multiplicacionMatricial(traspuesta(A), A))
    # Consigo los autovalores
    autovalores = diagonal(D)
    # Busco los indices del nucleo
    indices_nucleo = []
    n = len(autovalores)
    for i in range(n):
        if abs(autovalores[i]) <= tol:
            indices_nucleo.append(i)
    
    if not indices_nucleo:
        return np.array([])
    
    nucleo = C[:,indices_nucleo]

    if nucleo.shape[1] == 1:
        nucleo = nucleo.reshape((n, 1))

    return nucleo

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    dims = (m_filas,n_columnas)
    if listado == []:
        return {}, dims
    
    elementos = []
    for i in range(len(listado[2])):
        if not abs(listado[2][i]) <= tol:
            elementos.append(((listado[0][i],listado[1][i]), listado[2][i]))
    diccionario = dict(elementos)

    return diccionario, dims

def multiplica_rala_vector(A,v):
    dicc, (m, n) = A  # Desempaquetamos la tupla (diccionario, dimensiones)
    w = np.zeros(m)
    for (i, j), valor in dicc.items():
        w[i] += valor * v[j]
    return w

#%% Labo 08

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