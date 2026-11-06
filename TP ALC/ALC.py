import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt

import importlib
import ALC
import MODULO_ALC

from MODULO_ALC import rq, svd_reducida, calculaQR, multiplicacionMatricial, inversa, traspuesta, QR_con_HH, calculaCholesky, norma, productoExterno, productoInterno, calcularAx


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
    if (determinante3x3(M[:, :3]) < 0): #Si la matriz A = K*R tiene determinante negativa, le cambiamos el signo a la matriz ya que M ~ -M
        M = -M

    A = M[:, :3]                      # A sale de la M ya corregida
    b = M[:, 3]

    # 2) A = K0 R0 (K0 triangular superior, R0 ortogonal)
    K0, R0 = rq(A)

    #Sabemos que R es Ortogonal, osea que R*R^t = Id. Esto significa que R es una rotacion? No, 
    #det(R) = 1 y R*R^t = Id (esto seria una rotacion) | det(R) = -1 y R*R^t = Id (esto seria una reflexion)

    #Ahora veamos a K = [[a_u, y,u_0], Osea que debe ser triangular superior
    #                    [0, a_v, v_0],
    #                    [0,   0, 1]]
    #donde a_u y a_v son escalas de conversion entre las coordenadas de la camara y las coordenada de los pixeles, ambos valores deben ser mayor a 0
    #si fuesen negativos un punto que se desplaza hacia la derecha en el sistema de coordenadas de la camara se proyectaria hacia la izquierda en la imagen,
    #osea que tendriamos una reflexion del eje y no la transformacion que necesitamos.
    #Puede ser que la matriz K0 que obtenemos sea negativa en su diagonal. Por lo que necesitamos aplicar una correccion de signo, que nos sera util tanto para 
    #K1 como para R

    # 3) Diagonal de K positiva (S diagonal de +-1, S S = I, entonces K1 R = A)
    S = np.eye(3)
    for i in range(3):
        if (K0[i, i] < 0):
            S[i, i] = -1
    K1 = multiplicacionMatricial(K0, S)
    R = multiplicacionMatricial(S, R0)
    #Esto es valido porque como S^2 = Id y K1 * R = K1 * R0 = A, con esto obtenemos los tres elementos de la diagonal de k1 sea positivos
    #Y por proposicion det(AB) = det(A)*det(B), en nuestro caso det(A) = det(R)det(K1) y como R es ortogonal, det(R)=+- 1, pero det(A) es positiva y 
    # det(K1 tambien lo es, por lo tanto R debe tener determinante positiva y esto nos permitiria que la rotacion sea valida)

    # 4) Escala: M = s K [R|t], con s = K1[2,2] (la última fila de R tiene norma 1)
    escala = K1[2, 2]
    K = K1 / escala
    b = b.reshape(3, 1) / escala      # b/s = K t
    t = multiplicacionMatricial(inversa(K), b)

    return R, t.reshape(3), K

#Al correr esto deberia quedar algo como Q:(24x12), M:(3x4), K,R: (3x3) y t:(1x3)

#D
def devolverMatricesCalibracion3d(M, alphas_iguales=False, skew_cero=False):
    if (determinante3x3(M[:, :3]) < 0): #Si la matriz A = K*R tiene determinante negativa, le cambiamos el signo a la matriz ya que M ~ -M
        M = -M

    A = M[:, :3]                      # A sale de la M ya corregida
    b = M[:, 3]

    # 2) A = K0 R0 (K0 triangular superior, R0 ortogonal)
    K0, R0 = rq(A)
    # 3) Diagonal de K positiva (S diagonal de +-1, S S = I, entonces K1 R = A)
    S = np.eye(3)
    for i in range(3):
        if (K0[i, i] < 0):
            S[i, i] = -1
    K1 = multiplicacionMatricial(K0, S)
    R = multiplicacionMatricial(S, R0)

    # 4) Escala: M = s K [R|t], con s = K1[2,2] (la última fila de R tiene norma 1)
    escala = K1[2, 2]
    K = K1 / escala
    A_esc = A / escala                  #en caso de que lo necesitemos
    b = b.reshape(3, 1) / escala      # b/s = K t
    t = multiplicacionMatricial(inversa(K), b)

    #5)Impongo restricciones
    if (alphas_iguales == True):
        alpha = (K[0,0] + K[1,1]) / 2

        K[0,0] = alpha
        K[1,1] = alpha

    if (skew_cero == True):
        K[0,1] = 0

    #6) Recalculamos con svd
    if ((alphas_iguales == True) or (skew_cero == True)):
        C = multiplicacionMatricial(traspuesta(K),A_esc)
        U, matriz_sigmas, V_t = svd_reducida(C)
        D = np.eye(3)
        UV_t = multiplicacionMatricial(U, V_t)

        if (determinante3x3(UV_t) < 0):
            D[2,2] = -1

        R = multiplicacionMatricial(multiplicacionMatricial(U, D), V_t)

    else:
        R = multiplicacionMatricial(S, R0)

    t = multiplicacionMatricial(inversa(K), b)
    return R, t.reshape(3), K

#D
#Supongamos que corremos rq para obtener A = k0, r0, Pero esa K0 debe tener en k12 y = 0, pero si modificamos K0 con nuestras restricciones, debemos modificar R0 
#porque sino A != K0_modificado_con_restricciones * R0
#En otras palabras, despues de modificar a K0, debemos calcular la nueva R0_mod que sea ortogonal, para eso podemos usar SVD para encontrar la matriz de rotacion R0
#Mas cercana que minimice que ||A - K*R||^2 (norma_2), donde A ya deberia estar normalizada

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

        Q[2*i] = [U, V, 1, 0, 0, 0, -u*U, -u*V, -u]   # filas de Q pares (según ec. 11 del enunciado del tp).

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

    for H in Hlst:  
        h1 = H[:, 0]
        h2 = H[:, 1]

        v12 = amarv(h1, h2)
        v11 = amarv(h1, h1)
        v22 = amarv(h2, h2)

        filas_A.append(v12)
        filas_A.append(v11 - v22)

    A = np.array(filas_A)

    return A


def calcularMatrizK2D(A):
    _,_, V = svd_reducida(A)
    b = V[-1]

    #Sabemos que b = [b0, b1, b2, b3, b4, b5] y como B es simétrica entonces tenemos que:

    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]], 
                  [b[3], b[4], b[5]]
               ])

    if B[2, 2] < 0:
        B = -B

    L = calculaCholesky(B)
    K = inversa(traspuesta(L))
    K = K / K[2, 2] # Normalizamos para que K[2, 2] == 1

    return K

def obtenerPose2D(Hj, K):
    K_inv = inversa(K)
    R_t = multiplicacionMatricial(K_inv, Hj)  # R_t = K^-1 * H

    a1 = R_t[:, 0]
    a2 = R_t[:, 1]
    a3 = R_t[:, 2]

    escalar = 1 / norma(a1)

    # Forzamos que la profundidad sea positiva.
    if (a3[2] < 0):
        escalar = -escalar

    #Recuperamos los vectores de R (r1 y r2) y el de traslacion (t) 
    r1 = a1 * escalar
    r2 = a2 * escalar
    t = a3 * escalar

    r3 = productoExterno(r1, r2)  # r3 = r1 x r2

    # Rearmamos R y luego buscamos la mejor aproximación ortonormal usando SVD
    R = np.zeros((3,3))

    R[:, 0] = r1
    R[:, 1] = r2
    R[:, 2] = r3

    U,_, Vt = svd_reducida(R)
    R_ortogonal = productoExterno(U, Vt) #

    if determinante3x3(R_ortogonal) < 0:    #???
        R_ortogonal = -R_ortogonal
        t = -t

    return R_ortogonal, t

def normalizarHartley(pts):
    """
    Normalización isotrópica de Hartley para un conjunto de puntos 2D.

    Parámetros:
        pts: array de n x 2 con los puntos (u, v) o (U, V).

    Devuelve:
        pts_norm: array de n x 2 con los puntos normalizados (centroide en el
                  origen y distancia media al origen igual a sqrt(2)).
        T: matriz de 3 x 3 que lleva los puntos originales a los normalizados,
           en coordenadas homogéneas: p_norm = T * p.
    """
    # Centroide de los puntos: lo vamos a trasladar al origen
    centroide = np.mean(pts, axis=0)
    dif = pts - centroide

    # Distancia media de los puntos (ya centrados) al origen
    dist_media = np.mean(np.sqrt(np.sum(dif**2, axis=1)))

    # Factor de escala para que esa distancia media pase a ser sqrt(2)
    s = np.sqrt(2) / dist_media

    # Matriz de normalización: primero trasladamos (-centroide), después escalamos (s)
    T = np.array([[s, 0, -s*centroide[0]],
                  [0, s, -s*centroide[1]],
                  [0, 0, 1]])

    # Los puntos normalizados son los centrados y escalados
    return dif * s, T


def estimarH(imgPts, objPts):
    """
    Estima la homografía H que lleva puntos del plano del mundo a la imagen,
    usando DLT con normalización de Hartley.

    Parámetros:
        imgPts: array de n x 2 con los píxeles (u, v).
        objPts: array de n x 2 con los puntos del plano (U, V), en el mismo
                orden que imgPts (el punto i de uno corresponde al i del otro).

    Devuelve:
        H: matriz de 3 x 3 tal que (u, v, 1) ~ H * (U, V, 1), normalizada para
           tener norma de Frobenius igual a 1 (H está definida salvo escala).
    """
    # Normalizamos por separado los puntos del mundo y los de la imagen
    objN, Tw = normalizarHartley(objPts)
    imgN, Ti = normalizarHartley(imgPts)

    # Armamos Q (ecuación 11) con los puntos normalizados y resolvemos Qh = 0
    # tomando el último vector singular (mínimo de ||Qh|| con ||h|| = 1)
    Q = obtenerQ2d(imgN, objN)
    _, _, V = svd_reducida(Q)
    Hn = V[:, -1].reshape(3, 3)   # homografía en coordenadas normalizadas

    # Desnormalizamos (ecuación 14): H = Ti^-1 * Hn * Tw
    H = multiplicacionMatricial(inversa(Ti), multiplicacionMatricial(Hn, Tw))

    # H solo se conoce salvo escala: la normalizamos por su norma de Frobenius
    # (más estable que dividir por H[2, 2], que puede ser casi cero)
    return H / norma(H.reshape(-1), 2)


"""
------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Ejercicio 5 a):
def parametrosIntrinsecosEj5a():
    world_img_pts = pd.read_csv(r'\salguero\world_img_pts.csv')

    objPts = world_img_pts[["X", "Y"]].values
    H_lista = []

#Calculo H_0:
    imgPts_0 = world_img_pts[["u_0","v_0"]].values  # Extrae todas las filas de las columnas "u_0" y "v_0", osea matriz de (n,2)
    H_0 = estimarH(imgPts_0, objPts)
    H_lista.append(H_0)

#Calculo H_1:
    imgPts_1 = world_img_pts[["u_1","v_1"]].values  # Extrae todas las filas de las columnas "u_1" y "v_1", osea matriz de (n,2)
    H_1 = estimarH(imgPts_1,objPts)
    H_lista.append(H_1)

#Calculo H_2:
    imgPts_2 = world_img_pts[["u_2","v_2"]].values  # Extrae todas las filas de las columnas "u_2" y "v_2", osea matriz de (n,2)
    H_2 = estimarH(imgPts_2, objPts)
    H_lista.append(H_2)

#Calculo H_3:
    imgPts_3 = world_img_pts[["u_3","v_3"]].values  # Extrae todas las filas de las columnas "u_3" y "v_3", osea matriz de (n,2)
    H_3 = estimarH(imgPts_3, objPts)
    H_lista.append(H_3)


#Construimos la matriz A:
    A = armarA(H_lista)

#Obtenemos la matriz K con la matriz A:
    K = calcularMatrizK2D(A)

    return K


def poseYReproyeccionEj5b(archivo_txt,K,archivo_jpg):
    datos = np.loadtxt(archivo_txt)
    objPts = traspuesta(datos[1:3, :])  #Extraemos ejes X,Y de la foto 2d. Matriz de (51x2)
    imgPts = traspuesta(datos[4:6, :])  #Extraemos u, v que son las coordenadas en píxeles (u la columna y v la fila de la foto). Matriz de (51x2)

#Calculamos H (homografía) usando los datos del archivo txt:
    H = estimarH(imgPts, objPts)

#Calculamos la matriz R (matriz de rotacion que define la orientación 3d de la camara) y t (que es un vector que define la posición fisica 3d de la cámara) usando H y K:
    R, t = obtenerPose2D(H, K)



# ============================================================
# EJERCICIO 6: RESOLUCIÓN DEL ENUNCIADO
# ============================================================

# a)
# Se seleccionan el centro y los cuatro vértices exteriores de la H.
# El punto blanco identifica la esquina superior izquierda.
# Las correspondencias siguen el orden de object_points.csv.


# b)
def calculoLambdaMu(I0, J0):
    """
    Recibe los vectores 3D I0 y J0 del paso POS.
    Devuelve un par (lambda, mu); la otra solución es (-lambda, -mu).
    """
    # Restricciones: lambda² - mu² = a y 2*lambda*mu = b.
    a = productoInterno(J0, J0) - productoInterno(I0, I0)
    b = -2 * productoInterno(I0, J0)

    r = np.sqrt(a*a + b*b)

    # Los argumentos son no negativos; max evita negativos por redondeo.
    lbd = np.sqrt(max(0.0, (r + a) / 2))
    mu = np.sqrt(max(0.0, (r - a) / 2))

    if b < 0:
        mu = -mu

    # La otra solución se obtiene cambiando ambos signos.
    return lbd, mu


# c)
def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    """
    Recibe una solución Is, Js, la focal fx, la imagen de referencia
    m0 centrada y ajustada a fx, y el punto 3D de referencia M0.
    Devuelve i, j, k, Z0, R y t, con transformación objeto a cámara.
    """
    normaI = norma(Is, 2)
    normaJ = norma(Js, 2)

    # Recuperamos la profundidad y los ejes unitarios.
    Z0 = fx / normaI
    i = Is / normaI
    j = Js / normaJ

    # El tercer eje completa una base con orientación derecha.
    k = np.array([
        i[1]*j[2] - i[2]*j[1],
        i[2]*j[0] - i[0]*j[2],
        i[0]*j[1] - i[1]*j[0]
    ])

    R = np.array([i, j, k])

    # m0 está centrado y su coordenada vertical está ajustada a fx.
    T0 = np.array([
        Z0 * m0[0] / fx,
        Z0 * m0[1] / fx,
        Z0
    ])

    # T0 = R*M0 + t: convertimos la posición de referencia en traslación.
    RM0 = multiplicacionMatricial(
        R, M0.reshape((3, 1))
    ).reshape((3,))

    t = T0 - RM0

    return i, j, k, Z0, R, t


# d)
def evaluarPose(R, t, obj_vecs, img_vecs, K):
    """
    Recibe la pose objeto a cámara, los puntos 3D, sus correspondencias
    en píxeles originales y la matriz intrínseca K.
    Devuelve la cantidad de puntos delante de la cámara y su RMS en píxeles.
    """
    cantidad_validas = 0
    suma_errores = 0.0
    N = obj_vecs.shape[0]

    for idx in range(N):
        # Transformamos el punto del sistema del objeto al de la cámara.
        punto_cam = multiplicacionMatricial(
            R, obj_vecs[idx].reshape((3, 1))
        ).reshape((3,)) + t

        # Una proyección es válida si el punto está delante de la cámara.
        if punto_cam[2] > 0:
            cantidad_validas += 1

            q = multiplicacionMatricial(
                K, punto_cam.reshape((3, 1))
            ).reshape((3,))

            # Pasamos de coordenadas homogéneas a píxeles.
            u = q[0] / q[2]
            v = q[1] / q[2]

            du = u - img_vecs[idx, 0]
            dv = v - img_vecs[idx, 1]
            suma_errores += du*du + dv*dv

    # RMS de las proyecciones válidas, expresado en píxeles.
    if cantidad_validas > 0:
        rms = np.sqrt(suma_errores / cantidad_validas)
    else:
        rms = float("inf")

    return cantidad_validas, rms


# e)
def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    """
    Recibe puntos 3D coplanares y sus correspondencias 2D por filas,
    la matriz K sin sesgo, el máximo de iteraciones y la tolerancia del RMS.
    Devuelve R y t de la pose estimada, con transformación objeto a cámara.
    """
    if MaxIt < 1:
        raise ValueError("MaxIt debe ser al menos 1.")

    N = obj_vecs.shape[0]
    M0 = obj_vecs[0]

    fx = K[0, 0]
    fy = K[1, 1]
    cu = K[0, 2]
    cv = K[1, 2]

    # Cada fila contiene el desplazamiento desde M0 a otro punto.
    A = np.zeros((N - 1, 3))

    for idx in range(1, N):
        A[idx - 1] = obj_vecs[idx] - M0

    # Centramos los píxeles y expresamos ambas coordenadas con focal fx.
    # Esta conversión supone K[0, 1] = 0.
    imagen_posit = np.zeros((N, 2))

    for idx in range(N):
        imagen_posit[idx, 0] = img_vecs[idx, 0] - cu
        imagen_posit[idx, 1] = (
            (img_vecs[idx, 1] - cv) * fx / fy
        )

    m0 = imagen_posit[0]

    # A+ = V * Sigma^-1 * U.T; se calcula una sola vez.
    U, valores, V = svd_reducida(A)

    rango = valores.shape[0]
    if rango != 2:
        raise ValueError("Se requieren puntos coplanares de rango 2.")

    S_inv = np.zeros((rango, rango))

    for idx in range(rango):
        S_inv[idx, idx] = 1 / valores[idx]

    A_pinv = multiplicacionMatricial(
        multiplicacionMatricial(V, S_inv),
        U.T
    )

    # Las columnas de V generan el plano; su producto vectorial es normal.
    v1 = V[:, 0]
    v2 = V[:, 1]

    normal = np.array([
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ])
    normal = normal / norma(normal, 2)

    # La primera iteración usa la aproximación ortográfica escalada.
    eps = np.zeros(N - 1)
    rms_anterior = None

    for it in range(MaxIt):
        b_u = np.zeros(N - 1)
        b_v = np.zeros(N - 1)

        for idx in range(1, N):
            b_u[idx - 1] = (
                imagen_posit[idx, 0] * (1 + eps[idx - 1]) - m0[0]
            )
            b_v[idx - 1] = (
                imagen_posit[idx, 1] * (1 + eps[idx - 1]) - m0[1]
            )

        # Resolvemos los dos sistemas por mínimos cuadrados.
        I0 = multiplicacionMatricial(
            A_pinv, b_u.reshape((N - 1, 1))
        ).reshape((3,))

        J0 = multiplicacionMatricial(
            A_pinv, b_v.reshape((N - 1, 1))
        ).reshape((3,))

        lam, mu = calculoLambdaMu(I0, J0)

        mejor_rms = float("inf")
        mejor_pose = None

        # Evaluamos las dos soluciones del caso coplanar.
        for signo in [1, -1]:
            Is = I0 + signo * lam * normal
            Js = J0 + signo * mu * normal

            i, j, k, Z0, R, t = calcularSalidaPOSIT(
                Is, Js, fx, m0, M0
            )

            validas, rms = evaluarPose(
                R, t, obj_vecs, img_vecs, K
            )

            # Solo compiten poses con todos los puntos delante.
            if validas == N and rms < mejor_rms:
                mejor_rms = rms
                mejor_pose = (i, j, k, Z0, R, t)

        if mejor_pose is None:
            raise ValueError(
                "Ninguna pose tiene todos los puntos delante de la cámara."
            )

        i, j, k, Z0, R, t = mejor_pose

        # Detenemos si empeora o si la mejora es menor que atol.
        if rms_anterior is not None:
            mejora = rms_anterior - mejor_rms

            if mejora < 0:
                return R_anterior, t_anterior

            if mejora < atol:
                return R, t

        rms_anterior = mejor_rms
        R_anterior = R
        t_anterior = t

        # Actualizamos la perspectiva para la siguiente iteración.
        for idx in range(N - 1):
            eps[idx] = productoInterno(A[idx], k) / Z0

    return R, t


# CARGA DE DATOS PARA APLICAR EL MÉTODO

# Orden: centro, superior izquierda, superior derecha,
# inferior derecha, inferior izquierda.
img_vecs = np.array([
    [522.52398332, 1002.07768509],
    [400.73044838, 1095.50834202],
    [390.72002086,  943.68352450],
    [642.64911366,  910.31543274],
    [657.66475495, 1060.47184567]
])

datos = pd.read_csv(
    os.path.join("heli", "object_points.csv"),
    delimiter=";"
)
obj_vecs = np.asarray(datos[["U [m]", "V [m]", "W [m]"]])

K = np.loadtxt(
    os.path.join("heli", "K.csv"),
    delimiter=";"
)

MaxIt = 100
atol = 1e-6


# f)

# Estimamos la pose actual: P_camara = R_actual * P_objeto + t_actual.
R_actual, t_actual = aplicarPOSIT(
    obj_vecs, img_vecs, K, MaxIt, atol
)

# La posición de la cámara en el sistema de la H es C = -R.T * t.
C_actual = -multiplicacionMatricial(
    R_actual.T, t_actual.reshape((3, 1))
).reshape((3,))

# Elección: conservar la altura actual, pues el enunciado no fija otra.
# Convención: U hacia la derecha de la H, V hacia arriba y W hacia arriba del suelo.
h = C_actual[2]

R_deseada = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

C_deseada = np.array([0, 0, h])

t_deseada = -multiplicacionMatricial(
    R_deseada, C_deseada.reshape((3, 1))
).reshape((3,))

# Movimiento de la cámara expresado en los ejes de la H.
desplazamiento = C_deseada - C_actual

rotacion_movimiento = multiplicacionMatricial(
    R_deseada.T, R_actual
)

# Presentación de los resultados del inciso f).
print("Pose actual, objeto a cámara:")
print(R_actual)
print(t_actual)

print("Pose deseada, objeto a cámara:")
print(R_deseada)
print(t_deseada)

print("Desplazamiento en los ejes de la H [m]:")
print(desplazamiento)

print("Rotación del movimiento en los ejes de la H:")
print(rotacion_movimiento)
