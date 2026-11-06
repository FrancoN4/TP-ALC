import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt

import importlib
import alc

#%%
# a)
def obtenerQ3d(imgPts, objPts):
    """
    Recibe:
        imgPts: array (n, 2) con coordenadas de imagen (u, v).
        objPts: array (n, 3) con coordenadas reales (U, V, W).
        Las filas de ambos arrays deben corresponder al mismo punto.

    Devuelve:
        Q: matriz (2*n, 12) del sistema homogéneo DLT.
    """
    n = objPts.shape[0]
    Q = np.zeros((2*n, 12))

    for idx in range(n):
        U, V, W = objPts[idx]
        u, v = imgPts[idx]

        # Cada correspondencia aporta una ecuación para u y otra para v.
        Q[2*idx] = [
            U, V, W, 1,
            0, 0, 0, 0,
            -u*U, -u*V, -u*W, -u
        ]

        Q[2*idx + 1] = [
            0, 0, 0, 0,
            U, V, W, 1,
            -v*U, -v*V, -v*W, -v
        ]

    return Q


# b)
def resolverDLT3D(Q):
    """
    Recibe:
        Q: matriz (2*n, 12) del sistema DLT.

    Devuelve:
        M: matriz de proyección (3, 4), definida salvo escala.
    """
    # Minimizar ||Qm||² con ||m|| = 1 equivale a buscar un autovector
    # de Q.T*Q asociado a su menor autovalor.
    # Usamos diagRH para conservar también el posible vector del núcleo,
    # que la SVD reducida descarta cuando elimina valores singulares nulos.
    QtQ = multiplicacionMatricial(Q.T, Q)
    V, D = diagRH(QtQ)

    indice_min = 0

    for idx in range(1, D.shape[0]):
        if D[idx, idx] < D[indice_min, indice_min]:
            indice_min = idx

    # Los autovectores están almacenados como columnas.
    m = V[:, indice_min]

    # Las doce componentes de m representan las entradas de M por filas.
    return m.reshape((3, 4))


# c) y d)
def devolverMatricesCalibracion3d(
    M, alphas_iguales=False, skew_cero=False
):
    """
    Recibe:
        M: matriz de proyección (3, 4) de una cámara no degenerada.
        alphas_iguales: si es True, impone K[0, 0] = K[1, 1].
        skew_cero: si es True, impone K[0, 1] = 0.

    Devuelve:
        R: rotación (3, 3), de objeto a cámara.
        t: traslación de forma (3,), en las unidades de los puntos reales.
        K: matriz intrínseca (3, 3), normalizada con K[2, 2] = 1.

    Sin restricciones, recupera la descomposición de M salvo escala.
    Con restricciones, modifica K y calcula la rotación que minimiza
    ||A_esc - K*R||_F para esa K fija.
    """
    # M y -M representan la misma proyección.
    # Elegimos el signo compatible con K de diagonal positiva y det(R) = 1.
    if determinante3x3(M[:, :3]) < 0:
        M = -M

    A = M[:, :3]
    b = M[:, 3]

    # RQ separa A en una triangular superior y una matriz ortogonal.
    K0, R0 = rq(A)

    # Corregimos los signos sin alterar el producto:
    # (K0*S)*(S*R0) = K0*R0, porque S*S = I.
    S = np.eye(3)

    for idx in range(3):
        if K0[idx, idx] < 0:
            S[idx, idx] = -1

    K1 = multiplicacionMatricial(K0, S)
    R = multiplicacionMatricial(S, R0)

    # Normalizamos toda la matriz de proyección con la misma escala.
    escala = K1[2, 2]
    K = K1 / escala
    A_esc = A / escala
    b_esc = b.reshape((3, 1)) / escala

    # Imponemos una focal común mediante el promedio de las dos estimadas.
    if alphas_iguales:
        alpha = (K[0, 0] + K[1, 1]) / 2
        K[0, 0] = alpha
        K[1, 1] = alpha

    # Anulamos el sesgo entre los ejes de imagen.
    if skew_cero:
        K[0, 1] = 0

    if alphas_iguales or skew_cero:
        # Con K fija, minimizar ||A_esc - K*R||_F² equivale a
        # maximizar tr(R.T*C), donde C = K.T*A_esc.
        C = multiplicacionMatricial(K.T, A_esc)

        # Nuestra SVD devuelve U, valores singulares y V, no V.T.
        U, valores, V = svd_reducida(C)

        if valores.shape[0] != 3:
            raise ValueError(
                "El reajuste de la rotación requiere una matriz de rango 3."
            )

        # Procrustes ortogonal: R = U*D*V.T.
        # D evita obtener una reflexión en lugar de una rotación.
        UVt = multiplicacionMatricial(U, V.T)
        D = np.eye(3)

        if determinante3x3(UVt) < 0:
            # La SVD ordena los valores singulares de mayor a menor.
            D[2, 2] = -1

        R = multiplicacionMatricial(
            multiplicacionMatricial(U, D),
            V.T
        )

    # Recuperamos la traslación usando la K final: b_esc = K*t.
    t = multiplicacionMatricial(inversa(K), b_esc)

    return R, t.reshape((3,)), K


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def rq(M):
    """
    Recibe:
        M: matriz real (3, 3).

    Devuelve:
        K: matriz triangular superior (3, 3).
        R: matriz ortogonal (3, 3), tales que M = K*R.
    """
    # P invierte el orden de los ejes y cumple P.T = P y P*P = I.
    P = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    # Transformamos la descomposición RQ en una QR.
    M_mod = multiplicacionMatricial(
        P,
        multiplicacionMatricial(M.T, P)
    )

    Q_qr, R_qr = QR_con_HH(M_mod)

    # Deshacemos las transposiciones y las permutaciones.
    K = multiplicacionMatricial(
        P,
        multiplicacionMatricial(R_qr.T, P)
    )

    R = multiplicacionMatricial(
        P,
        multiplicacionMatricial(Q_qr.T, P)
    )

    return K, R


def determinante3x3(A):
    """
    Recibe:
        A: matriz (3, 3).

    Devuelve:
        Su determinante por expansión de la primera fila.
    """
    menor_00 = A[1, 1]*A[2, 2] - A[1, 2]*A[2, 1]
    menor_01 = A[1, 0]*A[2, 2] - A[1, 2]*A[2, 0]
    menor_02 = A[1, 0]*A[2, 1] - A[1, 1]*A[2, 0]

    return (
        A[0, 0]*menor_00
        - A[0, 1]*menor_01
        + A[0, 2]*menor_02
    )

#%%
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

#%%
"""
------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Ejercicio 5 a):
def parametrosIntrinsecosEj5a():
    world_img_pts = pd.read_csv(os.path.join("salguero", "world_img_pts.csv"))

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


#%%

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

#%%
###########################
#   MODULO ALC
##########################

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