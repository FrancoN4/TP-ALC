import numpy as np
import matplotlib.pyplot as plt
import random
from MODULO_ALC import rq, svd_reducida, calculaQR, multiplicacionMatricial, inversa, traspuesta, QR_con_HH, calculaCholesky, norma, productoExterno



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

        Q[2*i] = [U, V, 1, 0, 0, 0, U, V, 1, -u*U, -u*V, -u]   # filas de Q pares (según ec. 11 del enunciado del tp).

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
"""

#4

img = cv2.imread('futbol/test-enzo-gol.png')

img_rgb = img[:,:,::-1]

plt.figure(figsize=(12,8))
plt.imshow(img_rgb)
plt.grid(True, color='red', linestyle='--', linewidth=0.5)
plt.xlabel('u(pixeles horizontales)')
plt.ylabel('v(pixeles verticales)')
plt.title('Identificacion de pixeles (u,v)')
plt.show()

#a

Q = obtenerQ2d(imgPts, objPts)

U,S,V_t = svd_reducida(Q)

h = V_t[:,-1]
H = h.reshape(3,3)

H = H / H[2,2]

u = 810     #???
v = 520     #???

pt_real = productoExterno(inversa(H), np.array([u,v,1])) #transformamos el punto (x,v) a coordenadas reales (x,Y) usando la homografica H

x_real = pt_real[0]/pt_real[2]
y_real = pt_real[1]/pt_real[2]

dist_linea_meta = np.abs(x_real)
print(f"Coordenadas reales(X,Y):({x_real:.2f},{y_real:.2f})")
print(f"Distancia a la meta{dist_linea_meta:.2f}metros")
"""