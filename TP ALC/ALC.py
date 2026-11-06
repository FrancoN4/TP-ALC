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

# ============================================================
# EJERCICIO 3: CALIBRACIÓN CON PUNTOS COPLANARES
# ============================================================

def obtenerQ2d(imgPts, objPts):
    """
    Recibe:
        imgPts: coordenadas de imagen, de forma (n, 2).
        objPts: coordenadas del plano, de forma (n, 2).
        Las filas deben corresponder al mismo punto.

    Devuelve:
        Q: matriz de forma (2*n, 9) del sistema DLT.
    """
    n = objPts.shape[0]
    Q = np.zeros((2*n, 9))

    for idx in range(n):
        U, V = objPts[idx]
        u, v = imgPts[idx]

        # Cada correspondencia aporta dos ecuaciones.
        Q[2*idx] = [
            U, V, 1,
            0, 0, 0,
            -u*U, -u*V, -u
        ]

        Q[2*idx + 1] = [
            0, 0, 0,
            U, V, 1,
            -v*U, -v*V, -v
        ]

    return Q


def vectorMinimoDLT(A):
    """
    Recibe una matriz real A.
    Devuelve un vector unitario que minimiza ||A*x||_2.

    Utiliza el autovector de A.T*A asociado al menor autovalor,
    incluyendo los autovalores nulos.
    """
    AtA = multiplicacionMatricial(A.T, A)
    V, D = diagRH(AtA)

    indice_min = 0

    for idx in range(1, D.shape[0]):
        if D[idx, idx] < D[indice_min, indice_min]:
            indice_min = idx

    vector = V[:, indice_min]
    return vector / norma(vector, 2)


def normalizarHartley(pts):
    """
    Recibe:
        pts: puntos 2D de forma (n, 2).

    Devuelve:
        pts_norm: puntos centrados y escalados para que su
                  distancia media al origen sea sqrt(2).
        T: matriz homogénea que transforma los puntos originales
           en los normalizados.
    """
    centroide = np.mean(pts, axis=0)
    diferencias = pts - centroide

    distancias = np.sqrt(np.sum(diferencias**2, axis=1))
    distancia_media = np.mean(distancias)

    if distancia_media == 0:
        raise ValueError("No se pueden normalizar puntos coincidentes.")

    escala = np.sqrt(2) / distancia_media

    T = np.array([
        [escala, 0, -escala*centroide[0]],
        [0, escala, -escala*centroide[1]],
        [0, 0, 1]
    ])

    return diferencias * escala, T


def estimarH(imgPts, objPts):
    """
    Recibe:
        imgPts: puntos de imagen, de forma (n, 2).
        objPts: puntos del plano, de forma (n, 2).

    Devuelve:
        H: homografía del plano a la imagen, normalizada
           con norma de Frobenius igual a 1.
    """
    if imgPts.shape != objPts.shape or objPts.shape[0] < 4:
        raise ValueError(
            "Se requieren al menos cuatro correspondencias 2D."
        )

    # Normalizamos ambos conjuntos para mejorar la escala del sistema.
    objN, Tw = normalizarHartley(objPts)
    imgN, Ti = normalizarHartley(imgPts)

    Q = obtenerQ2d(imgN, objN)

    # Conservamos el vector del núcleo, incluso con solo cuatro puntos.
    h = vectorMinimoDLT(Q)
    Hn = h.reshape((3, 3))

    # Deshacemos las transformaciones de Hartley.
    H = multiplicacionMatricial(
        inversa(Ti),
        multiplicacionMatricial(Hn, Tw)
    )

    return H / norma(H.reshape((-1,)), 2)


def amarv(hi, hj):
    """
    Recibe dos columnas de una homografía, de forma (3,).
    Devuelve los seis coeficientes de la restricción hi.T*B*hj,
    donde B es simétrica.
    """
    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[0],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[2],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2]
    ])


def armarA(Hlst):
    """
    Recibe:
        Hlst: lista de homografías del plano a la imagen.

    Devuelve:
        A: matriz de forma (2*len(Hlst), 6) con las restricciones
           para estimar los parámetros intrínsecos.
    """
    filas = []

    for H in Hlst:
        h1 = H[:, 0]
        h2 = H[:, 1]

        # Ortogonalidad: h1.T*B*h2 = 0.
        filas.append(amarv(h1, h2))

        # Igual norma: h1.T*B*h1 - h2.T*B*h2 = 0.
        filas.append(amarv(h1, h1) - amarv(h2, h2))

    return np.array(filas)


def calcularMatrizK2D(A):
    """
    Recibe:
        A: matriz de restricciones de las homografías.

    Devuelve:
        K: matriz intrínseca triangular superior,
           normalizada con K[2, 2] = 1.

    Requiere que la B estimada sea definida positiva, salvo signo.
    """
    b = vectorMinimoDLT(A)

    # Orden de b: B11, B12, B22, B13, B23, B33.
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    # B y -B resuelven el mismo sistema homogéneo.
    if B[2, 2] < 0:
        B = -B

    # Reescalamos para facilitar las operaciones numéricas.
    if B[2, 2] == 0:
        raise ValueError("La matriz B estimada no permite recuperar K.")

    B = B / B[2, 2]

    # Comprobamos definición positiva mediante los menores principales.
    menor2 = B[0, 0]*B[1, 1] - B[0, 1]*B[1, 0]

    det_B = (
        B[0, 0]*(B[1, 1]*B[2, 2] - B[1, 2]*B[2, 1])
        - B[0, 1]*(B[1, 0]*B[2, 2] - B[1, 2]*B[2, 0])
        + B[0, 2]*(B[1, 0]*B[2, 1] - B[1, 1]*B[2, 0])
    )

    if B[0, 0] <= 0 or menor2 <= 0 or det_B <= 0:
        raise ValueError(
            "B no es definida positiva. Revisar las correspondencias, "
            "la diversidad de las vistas y la resolución del sistema."
        )

    # B es proporcional a K^-T*K^-1.
    # Si B = L*L.T, entonces K es proporcional a L^-T.
    L = calculaCholesky(B)
    K = inversa(L.T)

    return K / K[2, 2]


def obtenerPose2D(Hj, K):
    """
    Recibe:
        Hj: homografía del plano del objeto a la imagen.
        K: matriz intrínseca.

    Devuelve:
        R: rotación de objeto a cámara, de forma (3, 3).
        t: traslación de forma (3,), en las unidades de objPts.
    """
    columnas = multiplicacionMatricial(inversa(K), Hj)

    a1 = columnas[:, 0]
    a2 = columnas[:, 1]
    a3 = columnas[:, 2]

    # Las dos primeras columnas proporcionan estimaciones de la escala.
    denominador = norma(a1, 2) + norma(a2, 2)

    if denominador == 0:
        raise ValueError("La homografía no permite recuperar la pose.")

    escala = 2 / denominador

    # Elegimos el signo que sitúa el origen delante de la cámara.
    if a3[2] < 0:
        escala = -escala

    r1 = escala * a1
    r2 = escala * a2
    t = escala * a3

    # Tercera columna: producto vectorial r1 x r2.
    r3 = np.array([
        r1[1]*r2[2] - r1[2]*r2[1],
        r1[2]*r2[0] - r1[0]*r2[2],
        r1[0]*r2[1] - r1[1]*r2[0]
    ])

    R_aprox = np.zeros((3, 3))
    R_aprox[:, 0] = r1
    R_aprox[:, 1] = r2
    R_aprox[:, 2] = r3

    # Proyectamos la estimación sobre las matrices de rotación.
    # Tu SVD devuelve V, no V.T.
    U, valores, V = svd_reducida(R_aprox)

    if valores.shape[0] != 3:
        raise ValueError("La rotación aproximada no tiene rango 3.")

    UVt = multiplicacionMatricial(U, V.T)

    det_UVt = (
        UVt[0, 0]*(UVt[1, 1]*UVt[2, 2] - UVt[1, 2]*UVt[2, 1])
        - UVt[0, 1]*(UVt[1, 0]*UVt[2, 2] - UVt[1, 2]*UVt[2, 0])
        + UVt[0, 2]*(UVt[1, 0]*UVt[2, 1] - UVt[1, 1]*UVt[2, 0])
    )

    S = np.eye(3)

    # Corregimos una posible reflexión sin cambiar el signo de t.
    # La SVD ordena los valores singulares de mayor a menor.
    if det_UVt < 0:
        S[2, 2] = -1

    R = multiplicacionMatricial(
        multiplicacionMatricial(U, S),
        V.T
    )

    return R, t


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
