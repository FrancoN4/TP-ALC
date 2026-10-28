import numpy as np
from MODULO_ALC import rq, svd_reducida, calculaQR, multiplicacionMatricial, inversa

 
imgPts_todas = np.loadtxt('calib/3d/detected_corners.txt')
objPts = np.loadtxt('calib/3d/p_W_corners.txt', delimiter=',')
imgPts = imgPts_todas[0].reshape(12, 2)

#A
def obtenerQ3d(imgPts, objPts):

    n = objPts.shape[0]   # cantidad de esquinas de las hojas del mueble, osea hay 3 hojas ==> 3x4 = 12 esquinas ==> n = 12.
    Q = np.zeros ((2*n, 12)) # matriz de 24x12 (tamaño de Q según ec. 6 del enunciado del tp).

    for i in range(0,n):
        U, V, W = objPts[i]   # coordenadas del sist. de referencia de la vida real para las esquinas. "objPts[i]" es una tupla de (U_i, V_i, W_i).
        u, v = imgPts[i]   # par de coordenadas de la imagen del objeto (2d). "imgPts[i]" es una tupla de 24 elems donde (u_1, v_1, u_2, v_2, .... , u_n, v_n), con n = 12 (cant de esquinas)

        Q[2*i] = [U, V, W, 1, 0, 0, 0, 0, -u*U, -u*V, -u*W, -u]   # filas de Q pares (según ec. 6 del enunciado del tp).

        Q[2*i + 1] = [0, 0, 0, 0, U, V, W, 1, -v*U, -v*V, -v*W, -v] # filas de Q impares (según ec.6 del enunciado del tp).

    return Q

#A
def resolverDLT3D(Q):
    U, S, Vt = svd_reducida(Q)  # Descomposición en valores singulares de Q
    M = Vt[-1,:].reshape(3, 4)  # Tomamos la última fila de Vt y la reestructuramos en una matriz 3x4
    return M

#C
def devolverMatricesCalibracion3d(M):
    M3x3 = M[:, :3]  # Extraemos la submatriz 3x3 de M
    K0, R = rq(M3x3)  # Obtenemos K y R usando la función rq
    escala = K0[2, 2]

    K = K0 / escala
    m4 = M[:, 3] / escala

    t = multiplicacionMatricial(inversa(K), m4) #calculamos el vector de traslacion t

    return R, t, K

#D
