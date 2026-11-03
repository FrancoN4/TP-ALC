import numpy as np
import matplotlib.pyplot as plt
import random
from MODULO_ALC import rq, svd_reducida, calculaQR, multiplicacionMatricial, inversa, traspuesta, QR_con_HH

#A
def obtenerQ3d(imgPts, objPts):

    n = objPts.shape[0]   # cantidad de esquinas de las hojas del mueble, osea hay 3 hojas ==> 3x4 = 12 esquinas ==> n = 12.
    Q = np.zeros ((2*n, 12)) # matriz de 24x12 (tamaño de Q según ec. 6 del enunciado del tp).

    for i in range(n):
        U, V, W = objPts[i]   # coordenadas del sist. de referencia de la vida real para las esquinas. "objPts[i]" es una tupla de (U_i, V_i, W_i).
        u, v = imgPts[i]   # par de coordenadas de la imagen del objeto (2d). "imgPts[i]" es una tupla de 24 elems donde (u_1, v_1, u_2, v_2, .... , u_n, v_n), con n = 12 (cant de esquinas)

        Q[2*i] = [U, V, W, 1, 0, 0, 0, 0, -u*U, -u*V, -u*W, -u,]   # filas de Q pares (según ec. 6 del enunciado del tp).
        Q[2*i+1] = [ 0, 0, 0, 0, U, V, W, 1, -v*U, -v*V, -v*W, -v]
        
    return Q

#B
def resolverDLT3D(Q):
    U, S, Vt = svd_reducida(Q)  # Descomposición en valores singulares de Q
    M = Vt[-1,:].reshape(3, 4)  # Tomamos la última fila de Vt y la reestructuramos en una matriz 3x4
    return M

#C
def devolverMatricesCalibracion3d(M):
    if determinante3x3(M[:, :3]) < 0:
        M = -M

    A = M[:, :3]                      # A sale de la M ya corregida
    b = M[:, 3]

    # 2) A = K0 R0 (K0 triangular superior, R0 ortogonal)
    K0, R0 = rq(A)

    # 3) Diagonal de K positiva (S diagonal de +-1, S S = I, entonces K1 R = A)
    S = np.eye(3)
    for i in range(3):
        if K0[i, i] < 0:
            S[i, i] = -1
    K1 = multiplicacionMatricial(K0, S)
    R = multiplicacionMatricial(S, R0)

    # 4) Escala: M = s K [R|t], con s = K1[2,2] (la última fila de R tiene norma 1)
    escala = K1[2, 2]
    K = K1 / escala
    b = b.reshape(3, 1) / escala      # b/s = K t
    t = multiplicacionMatricial(inversa(K), b)

    return R, t.reshape(3), K

#D

#aux
def rq(M):
    #Recibe una matriz de 3x3 y devuelve las matrices K y R de la descomposición RQ.
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    M_mod = multiplicacionMatricial(P, multiplicacionMatricial(traspuesta(M), P))

    Q_qr, R_qr = QR_con_HH(M_mod)

    K = multiplicacionMatricial(P, multiplicacionMatricial(traspuesta(R_qr), P))
    R = multiplicacionMatricial(P, multiplicacionMatricial(traspuesta(Q_qr), P))

    return K, R

def determinante3x3(A):
    return (A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) - A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0]) + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0]))

#############################################################################################
#Ejercicio 3
"""Ejercicio 3 a)
Sabemos que la matriz H es de 3x3, y que si lo vectorizamos, obtenemos h, una matriz de 9x1.
También sabemos que Q * h = 0 para Q matriz de (2n)x9
Además, por el enunciado del tp, tenemos que lambda[u, v, 1]^T = H[U, V, 1]^T


Como cada punto aporta 2 filas, se necesitan como mínimo n = 4 puntos para disponer de 8 filas linealmente independiente.
Si n > 4, la eliminación gaussiana convierte en ceros las demás filas debido a que la matriz al tener 9 columnas, es imposible que haya mas de esa cantidad de filas l.i.


Además, si en vez de 8 filas l.i, tuviera 9 (matriz cuadrada), no habría tal variable libre y la única solución de Qh = 0 sería h=0, lo cual la matriz H sería igual a 0.
Como eso no puede pasar pues no tiene sentido físico, el numero máximo de filas l.i permitido de Q es 8.
===> rango(Q) = 8.

 
Dado que H es una matriz que, según la definición del principio, las matrices H y lambdaH representan exactamente la misma proyección 2D para cualquier lambda != 0, todas las soluciones de la forma lambda*h_0 corresponden a la misma y única homografía física H.
Entonces nos garantiza de esta forma que H tiene solución única a menos de un factor de escala.
"""
objPts = np.zeros((77, 2))

indice = 0

for j in range(7):
    for i in range(11):
        objPts[indice] = [30*i, 30*j]
        indice += 1



def obtenerQ2d(imgPts, objPts):

    n = objPts.shape[0]   # cantidad de esquinas internas del tablero.
    Q = np.zeros((2*n, 9)) # matriz de (2n)x9 (tamaño de Q según ec. 11 del enunciado del tp).

    for i in range(0,n):
        U, V = objPts[i]   # coordenadas del sist. de referencia de la vida real para las esquinas internas del tablero (matriz de 11x7) (2d). "objPts[i]" es una tupla de (U_i, V_i).
        u, v = imgPts[i]   # par de coordenadas de la imagen del objeto (2d).

        Q[2*i] = [U, V, 1, 0, 0, 0, 0, 0, 0, U, V, 1, -u*U, -u*V, -u, -v*U, -v*V, -v]   # filas de Q pares (según ec. 11 del enunciado del tp).

        Q[2*i + 1] = [0, 0, 0, U, V, 1, -v*U, -v*V, -v] # filas de Q impares (según ec. 11 del enunciado del tp).

    return Q


def amarv(hi, hj):

    v_ij = np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2]
        ])

    return v_ij


def armarA(Hlst):
    filas_A = []

    for H in lista_H:   #implementacion del jupiter
        h1 = H[:, 0]
        h2 = H[:, 1]

        v12 = amarv(h1, h2)
        v11 = amarv(h1, h1)
        v22 = amarv(h2, h2)

        filas_A.append(v12)
        filas_A.append(v11 - v22)

    A = np.array(filas_A)

    return A
"""
def calcularMatrizK2D(A):
    , , V = alc.svd(A)
    b = V[-1]

    Sabemos que b = [b0, b1, b2, b3, b4, b5] y como B es simétrica entonces tenemos que:

    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]], 
                  [b[3], b[4], b[5]]
               ])

    if B[2, 2] < 0:
        B = -B

    L = alc.calculaCholesky(B)
    K = alc.inversa(alc.traspuesta(L))
    K = K / K[2, 2] # Normalizamos para que K[2, 2] == 1

    return K

def obtenerPose2D(Hj, K):
    K_inv = alc.inversa(K)
    R_t = alc.prodMat(K_inv, Hj)  # R_t = K^-1 * H

    a1 = R_t[:, 0]
    a2 = R_t[:, 1]
    a3 = R_t[:, 2]

    escalar = 1 / alc.norma(a1)

    r1 = a1 * escalar
    r2 = a2 * escalar
    t = a3 * escalar

    r3 = alc.prodVectorial(r1, r2)  # r3 = r1 x r2

    # Rearmamos R y luego buscamos la mejor aproximación ortonormal usando SVD
    R = np.columnstack((r1, r2, r3))
    U, , Vt = alc.svd(R)
    R_ortogonal = alc.prodMat(U, Vt)

    if alc.det(R_ortogonal) < 0:
        R_ortogonal = -R_ortogonal
        t = -t

    return R_ortogonal, t
"""