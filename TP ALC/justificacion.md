## Ejercicio 1 (a): DLT en 3 dimensiones

### Construcción de la matriz Q (Introduccion)

Podemos observar que en `detected_corners.txt` cada fila corresponde a una imagen. Como cada punto detectado tiene dos coordenadas, $(u_i,v_i)$, y cada fila contiene 24 columnas, significa que tenemos 12 puntos detectados por imagen. Por lo tanto, podemos representar cada fila de la siguiente manera:

$$
[u_0,v_0,u_1,v_1,\ldots,u_{11},v_{11}]
$$

**Ecuaciones de una correspondencia.** Para cada correspondencia 3D-2D ($i=0,\ldots,11$) tenemos un punto del objeto, en coordenadas homogéneas:

$$
P_i=(U_i,V_i,W_i,1)^\top
$$

y su píxel correspondiente $(u_i,v_i)$. La ecuación (3) del enunciado dice:

$$
\lambda_i
\begin{bmatrix}
u_i\\
v_i\\
1
\end{bmatrix}
=MP_i,
\qquad
M=
\begin{bmatrix}
m_1^\top\\
m_2^\top\\
m_3^\top
\end{bmatrix}
\in\mathbb R^{3\times4}.
$$

Desarrollando el producto fila por fila (cada $m_j^\top P_i$ es un número), obtenemos:

$$
m_1^\top P_i=\lambda_i u_i,
\qquad
m_2^\top P_i=\lambda_i v_i,
\qquad
m_3^\top P_i=\lambda_i.
$$

$\lambda_i$ es el factor de escala de cada punto (es proporcional a su profundidad), por lo que es una incógnita distinta para cada correspondencia. Para no agregar una incógnita por punto, la eliminamos reemplazando la tercera ecuación en las otras dos:

$$
(m_1^\top-u_im_3^\top)P_i=0,
\qquad
(m_2^\top-v_im_3^\top)P_i=0.
$$

Estas ecuaciones son **lineales** en las 12 entradas de $M$, porque $u_i$, $v_i$ y $P_i$ son datos conocidos.

Apilamos las filas de $M$, escritas como columnas, en el vector:

$$
m=
\begin{bmatrix}
m_1\\
m_2\\
m_3
\end{bmatrix}
\in\mathbb R^{12}.
$$

Como un número es igual a su transpuesto, $m_j^\top P_i=P_i^\top m_j$, y entonces

$$
(m_1^\top-u_im_3^\top)P_i=P_i^\top m_1+0^\top m_2-u_iP_i^\top m_3=
\begin{bmatrix}P_i^\top&0^\top&-u_iP_i^\top\end{bmatrix}m .
$$

Haciendo lo mismo con la ecuación de $v_i$, cada punto aporta dos filas de $Q$ (ecuaciones 5 y 6 del enunciado):

$$
\begin{bmatrix}
U_i&V_i&W_i&1&0&0&0&0&
-u_iU_i&-u_iV_i&-u_iW_i&-u_i\\
0&0&0&0&U_i&V_i&W_i&1&
-v_iU_i&-v_iV_i&-v_iW_i&-v_i
\end{bmatrix}
m=
\begin{bmatrix}
0\\
0
\end{bmatrix}.
$$

Entonces, como tenemos 12 puntos por imagen y cada uno aporta dos ecuaciones, obtenemos:

$$
Q\in\mathbb R^{24\times12}.
$$

En otras palabras, tendremos 24 ecuaciones y 12 incógnitas por cada imagen.

**¿La matriz de proyección $M_j$ es igual para todas las imágenes?**

No. Cada imagen tiene una posición y orientación relativas diferentes entre la cámara y el objeto. Por lo tanto, sus parámetros extrínsecos, es decir, la matriz de rotación $R_j$ y el vector de traslación $t_j$, también son diferentes.

Sin embargo, como estamos utilizando la misma cámara, sus parámetros intrínsecos $K$ permanecen constantes (suponiendo que no se modifica la configuración de la cámara entre capturas).

Por lo tanto, para cada imagen $j$ tenemos una matriz de proyección diferente:

$$
\boxed{M_j=K[R_j\mid t_j]}
$$

En consecuencia, debemos construir una matriz $Q_j$ independiente para cada imagen, ya que cada una permite estimar su propia matriz de proyección $M_j$.

---

### ¿Qué rango debe tener Q para obtener una solución no nula única, salvo escala?

Queremos determinar el rango que debe tener la matriz $Q$ para que el sistema homogéneo:

$$
Qm=0
$$

tenga una solución no nula única, salvo un factor de escala. Sabemos que:

$$
Q\in\mathbb R^{2n\times12}
$$

(con $n$ la cantidad de puntos) y que:

$$
m\in\mathbb R^{12}.
$$

Recordemos las definiciones:

$$
\ker(Q)=\{x\in\mathbb R^{12}:Qx=0\},
$$

$$
\operatorname{im}(Q)=
\{b\in\mathbb R^{2n}:b=Qx,\text{ para algún }x\in\mathbb R^{12}\},
$$

$$
\operatorname{rango}(Q)=\dim(\operatorname{im}(Q)).
$$

Las soluciones del sistema $Qm=0$ son, por definición, los vectores de $\ker(Q)$.

Para que exista una única solución no nula salvo escala, todas las soluciones deben ser múltiplos de un mismo vector, es decir:

$$
\dim(\ker(Q))=1.
$$

**Teorema de la dimensión.** Dada una matriz $A\in\mathbb K^{p\times q}$ (usamos $p\times q$ en lugar de $m\times n$ para no pisar las letras $m$, vector de incógnitas, y $n$, cantidad de puntos), se cumple que:

$$
q=\dim(\ker(A))+\dim(\operatorname{im}(A)),
$$

donde $q$ es la cantidad de columnas de $A$.

Nuestra matriz $Q$ tiene 12 columnas, entonces:

$$
12=\dim(\ker(Q))+\dim(\operatorname{im}(Q))
$$

$$
12=\dim(\ker(Q))+\operatorname{rango}(Q).
$$

Como necesitamos que $\dim(\ker(Q))=1$:

$$
12=1+\operatorname{rango}(Q)
$$

Por lo tanto:

$$
\boxed{\operatorname{rango}(Q)=11}
$$

**¿Qué pasa si el rango no es 11?**

- Si $\operatorname{rango}(Q)<11$, entonces $\dim(\ker(Q))\geq2$. El conjunto de soluciones es un subespacio de dimensión al menos 2: existen dos soluciones linealmente independientes y, por lo tanto, infinitas soluciones no proporcionales entre sí. Hay más de una variable libre y no se puede determinar $M$ de manera única salvo escala.

- Si $\operatorname{rango}(Q)=12$, entonces $\ker(Q)=\{0\}$. La única solución es $m=0$, que no representa ninguna cámara.

**Observación (datos exactos vs. datos reales).**

Con datos exactos, el vector $m$ verdadero (no nulo) pertenece a $\ker(Q)$, por lo que:

$$
\operatorname{rango}(Q)\leq11.
$$

Lo que necesitamos asegurar es que sea exactamente 11.

Con datos reales (errores al marcar los píxeles, ruido, errores al medir las coordenadas del objeto, etc.), $Q$ suele tener rango 12 y la única solución exacta de $Qm=0$ es $m=0$.

Por eso, en el inciso (b), la igualdad exacta se reemplaza por minimizar:

$$
\boxed{
\min_{\|m\|_2=1}\|Qm\|_2^2
}
$$

como se plantea en la ecuación 7 del enunciado.

---

### ¿Cuántos puntos se necesitan como mínimo?

Sabemos que:

$$
Q\in\mathbb R^{2n\times12}
$$

y que necesitamos:

$$
\operatorname{rango}(Q)=11.
$$

La imagen de una matriz está generada por sus columnas, porque:

$$
Qx=x_1c_1+x_2c_2+\dots+x_{12}c_{12},
$$

donde $c_1,\dots,c_{12}$ son las columnas de $Q$: todo vector de $\operatorname{im}(Q)$ es una combinación lineal de ellas.

Además, $\operatorname{im}(Q)$ es un subespacio de $\mathbb R^{2n}$.

De ahí salen dos cotas para el rango:

$$
\operatorname{rango}(Q)
=\dim(\operatorname{im}(Q))
\leq 2n
\quad
(\text{subespacio de }\mathbb R^{2n}),
$$

$$
\operatorname{rango}(Q)\leq12
\quad
(\text{generado por 12 columnas}).
$$

Para que el rango pueda alcanzar el valor 11, necesariamente debe cumplirse:

$$
2n\geq11
\quad\Longrightarrow\quad
n\geq\frac{11}{2}.
$$

Como $n$ es una cantidad de puntos, debe ser entero. Por lo tanto:

$$
\boxed{n_{\min}=6\text{ puntos}}
$$

Por ejemplo, con $n=5$ tendríamos 10 ecuaciones, entonces:

$$
\operatorname{rango}(Q)\leq10
$$

y, por el teorema de la dimensión:

$$
\dim(\ker(Q))\geq2.
$$

Por lo tanto, no habría unicidad.

Es importante aclarar que 6 puntos son el mínimo necesario por cantidad de ecuaciones, pero no garantizan por sí solos una solución única. Los puntos también deben cumplir ciertas condiciones geométricas.

---

### ¿Qué condición geométrica deben cumplir los puntos?

Para que $Q$ tenga rango 11 no alcanza con tener $2n\geq11$ filas: la imagen de $Q$, generada por sus 12 columnas, tiene que tener dimensión 11. Si los puntos están mal ubicados, hay columnas que no aportan una dirección nueva (por ejemplo, columnas nulas) y el rango baja.

**Condición necesaria: los puntos no pueden ser todos coplanares.**

Veamos qué pasa si lo están. Existe un cambio de coordenadas del objeto $P_i'=TP_i$, con $T\in\mathbb R^{4\times4}$ invertible, tal que ese plano pasa a ser $W=0$. Los píxeles no cambian, y las filas de la nueva matriz $Q'$ son las de $Q$ multiplicadas a derecha por la matriz $\operatorname{diag}(T^\top,T^\top,T^\top)\in\mathbb R^{12\times12}$, que es invertible. Por lo tanto $\operatorname{rango}(Q')=\operatorname{rango}(Q)$, y alcanza con analizar el caso $W=0$.

Entonces:

$$
P_i=(U_i,V_i,0,1)^\top
$$

para todo $i$, y las dos filas de cada punto quedan:

$$
\begin{bmatrix}
U_i&V_i&\mathbf{0}&1&0&0&\mathbf{0}&0&
-u_iU_i&-u_iV_i&\mathbf{0}&-u_i\\
0&0&\mathbf{0}&0&U_i&V_i&\mathbf{0}&1&
-v_iU_i&-v_iV_i&\mathbf{0}&-v_i
\end{bmatrix}.
$$

Las columnas 3, 7 y 11 (las que multiplican a $W_i$) son **idénticamente nulas**, para todos los puntos.

Entonces, $\operatorname{im}(Q)$ está generada por las otras 9 columnas, y:

$$
\operatorname{rango}(Q)
=\dim(\operatorname{im}(Q))
\leq9<11.
$$

Por el teorema de la dimensión:

$$
\dim(\ker(Q))=12-\operatorname{rango}(Q)\geq3.
$$

En otras palabras, las incógnitas $m_{13}$, $m_{23}$ y $m_{33}$ ni siquiera aparecen en las ecuaciones: sus valores no pueden determinarse a partir de estas correspondencias.

**Interpretación:** si todos los puntos están en $W=0$, los datos no dicen nada sobre cómo la cámara trata la coordenada $W$, que es justamente la que hace falta para describir un objeto en 3D. Las 9 columnas restantes forman el sistema de la homografía, que se resuelve en el Ejercicio 3.

**Conclusión.** Para poder obtener:

$$
\operatorname{rango}(Q)=11
$$

necesitamos, como mínimo:

$$
\boxed{
n\geq6
\qquad\text{y}\qquad
\text{los puntos no son todos coplanares}.
}
$$

Estas son condiciones necesarias, pero no suficientes por sí solas: también hay que evitar otras configuraciones geométricas degeneradas. Por eso, en la práctica, los puntos deben estar bien distribuidos en el volumen del objeto y se utilizan más de 6 correspondencias.

**Verificación en nuestros datos.** En nuestro caso las 12 esquinas están repartidas en 3 hojas del mueble. Cada hoja es plana, pero eso no alcanza para afirmar que el conjunto completo no es coplanar, así que lo comprobamos. Sea $\mathcal P\in\mathbb R^{12\times4}$ la matriz cuyas filas son los $P_i^\top$ (los puntos de `p_W_corners.txt` en coordenadas homogéneas). Los puntos son coplanares si y sólo si existe $\pi\neq0$ con $\mathcal P\pi=0$, es decir, si y sólo si $\operatorname{rango}(\mathcal P)<4$. Entonces basta verificar que $\operatorname{rango}(\mathcal P)=4$, o sea que su cuarto valor singular no sea del orden del error de máquina.

Con los datos reales, por el ruido, el rango numérico de $Q$ es 12. Lo que esperamos observar es que $\sigma_{12}$ sea mucho menor que $\sigma_{11}$ (rango "efectivo" 11), y por eso en el inciso (b) se resuelve la minimización de la ecuación 7.

--

## Ejercicio 1 (b): Encontrar el vector $m$ que resuelve la minimización

Partimos de la ecuación 8 del enunciado:

$$
\boxed{
\min_{\|m\|_2=1}m^\top Q^\top Qm
}
$$

Primero, recordemos que:

$$
m^\top Q^\top Qm=(Qm)^\top(Qm)=\|Qm\|_2^2
$$

Por lo tanto, podemos expresar el problema como:

$$
\boxed{
\min_{\|m\|_2=1}\|Qm\|_2^2
}
$$

La restricción $\|m\|_2=1$ evita que la solución trivial $m=0$ sea el mínimo.


### Resolución mediante SVD

**Definición 7.1 (SVD).** Dada $A\in\mathbb R^{p\times q}$ (usamos $p\times q$ en lugar de $m\times n$ para no pisar la letra $m$ del vector de incógnitas), la descomposición en valores singulares es una factorización

$$
A=U\Sigma V^\top,
$$

con $U\in\mathbb R^{p\times p}$ y $V\in\mathbb R^{q\times q}$ matrices ortogonales (Definición 3.7; como trabajamos en $\mathbb R$, $V^*=V^\top$) y $\Sigma\in\mathbb R^{p\times q}$. Si $p\geq q$, la matriz $\Sigma$ tiene la forma

$$
\Sigma=\begin{pmatrix}D\\ 0_{(p-q)\times q}\end{pmatrix},
\qquad
D=\operatorname{diag}(\sigma_1,\dots,\sigma_q),
\qquad
\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_q\geq0 .
$$

Los $\sigma_i$ son los valores singulares de $A$ y las columnas de $V$ son sus vectores singulares derechos.

Como queremos minimizar la norma de $Qm$, aplicamos esta definición a $A=Q$, que tiene más filas que columnas ($24\geq12$):

$$
Q=U\Sigma V^\top
$$

Donde, para nuestro caso, tenemos:

$$
Q\in\mathbb R^{24\times12},
\qquad
U\in\mathbb R^{24\times24},
\qquad
\Sigma=\begin{pmatrix}D\\ 0_{12\times12}\end{pmatrix}\in\mathbb R^{24\times12},
\qquad
V\in\mathbb R^{12\times12}.
$$

Reemplazamos la descomposición SVD en nuestra función objetivo. Como $U$ es ortogonal, se cumple que $U^\top U=I$ (Definición 3.7). Entonces:

$$
\begin{aligned}
\|Qm\|_2^2
&=(U\Sigma V^\top m)^\top(U\Sigma V^\top m)\\
&=m^\top V\Sigma^\top U^\top U\Sigma V^\top m\\
&=m^\top V\Sigma^\top\Sigma V^\top m.
\end{aligned}
$$

Además, por la forma de $\Sigma$, resulta $\Sigma^\top\Sigma=D^\top D=D^2$ (el bloque de ceros no aporta nada).

Ahora, definimos:

$$
z=V^\top m
$$

Como $V$ es ortogonal, también lo es $V^\top$ (Observación 3.3), y se cumple $VV^\top=I$. Entonces:

$$
\|z\|_2^2=m^\top VV^\top m=\|m\|_2^2=1.
$$

Por lo tanto, nuestra función objetivo queda:

$$
\begin{aligned}
\|Qm\|_2^2
&=z^\top D^2 z\\
&=\sigma_1^2z_1^2+\sigma_2^2z_2^2+\cdots+\sigma_{12}^2z_{12}^2,
\end{aligned}
\qquad\text{con}\qquad
z_1^2+z_2^2+\cdots+z_{12}^2=1.
$$

Como los valores singulares están ordenados de mayor a menor, $\sigma_i\geq\sigma_{12}$ para todo $i$, y entonces:

$$
\sigma_1^2z_1^2+\cdots+\sigma_{12}^2z_{12}^2
\;\geq\;
\sigma_{12}^2\,(z_1^2+\cdots+z_{12}^2)
=\sigma_{12}^2 .
$$

Es decir, $\|Qm\|_2^2\geq\sigma_{12}^2$, y la cota se alcanza cuando toda la contribución corresponde al menor valor singular:

$$
z=
\begin{bmatrix}
0\\
0\\
\vdots\\
0\\
\pm1
\end{bmatrix}.
$$

Como habíamos definido $z=V^\top m$, multiplicando por $V$ obtenemos $m=Vz$ (porque $VV^\top=I$). Por lo tanto:

$$
\boxed{m=\pm v_{12}}
$$

Donde $v_{12}$ es la última columna de $V$, correspondiente al menor valor singular de $Q$. Si $\sigma_{11}>\sigma_{12}$, estos son los únicos mínimos: la solución es única salvo el signo.

### Recuperación de la matriz de proyección M

Una vez obtenido el vector $m$, podemos reconstruir la matriz de proyección $M$.

Recordemos que habíamos definido $m=(m_1;m_2;m_3)$, es decir, las filas de $M$ una detrás de otra:

$$
m=
\begin{bmatrix}
m_{11}&m_{12}&m_{13}&m_{14}&m_{21}&m_{22}&m_{23}&m_{24}&m_{31}&m_{32}&m_{33}&m_{34}
\end{bmatrix}^\top.
$$

Por lo tanto, reorganizando sus componentes obtenemos:

$$
\boxed{
M=
\begin{bmatrix}
m_{11}&m_{12}&m_{13}&m_{14}\\
m_{21}&m_{22}&m_{23}&m_{24}\\
m_{31}&m_{32}&m_{33}&m_{34}
\end{bmatrix}
}
$$

### Conclusión

La solución de la minimización de la ecuación 8 se obtiene directamente mediante la descomposición SVD de $Q$.

Por lo tanto, debemos calcular:

$$
Q=U\Sigma V^\top
$$

Y seleccionar la última columna de $V$, suponiendo que los valores singulares están ordenados de mayor a menor:

$$
\boxed{m=\pm V[:,11]}
$$

(índices contando desde 0). Finalmente, reorganizamos las 12 componentes de $m$ para obtener la matriz de proyección $M$ de dimensión $3\times4$.

La restricción $\|m\|_2=1$ fija la escala de $m$, así que la solución queda determinada salvo el signo. En cuanto a $M$, queda definida a menos de signo y escala, como corresponde al modelo de proyección perspectiva, y ambos se corrigen al descomponerla en $K$, $R$ y $t$ (Ejercicio 2c).


# Ejercicio 3(a): Estimación de homografías mediante DLT

## 1. Introducción

Ahora trabajamos con puntos coplanares, es decir, puntos que pertenecen a un mismo plano.

Disponemos de un tablero de ajedrez de $12\times 8$ cuadrados, por lo que tiene $11\times 7$ esquinas internas. Cada cuadrado mide $30\,\text{mm}\times 30\,\text{mm}$, y del cual tomamos como sistema de referencia del tablero un origen en una de sus esquinas internas, con los ejes $U$ y $V$ a lo largo de sus lados y $W=0$ (ya que todos los puntos pertenecen a un mismo plano).

Por lo tanto, las coordenadas de las esquinas internas son:

$$
U=30i,\qquad V=30j,\qquad W=0
$$

donde:

$$
i=0,\ldots,10,\qquad j=0,\ldots,6
$$

En total, tenemos:

$$
n=11\cdot 7=77
$$

correspondencias entre puntos del tablero y sus posiciones en la imagen.

Queremos estimar una homografía $H_j$ para cada una de las 20 fotografías del tablero, utilizando el método DLT.

---

## 2. Planteo del sistema

Como $W=0$, la tercera columna de $R$ queda multiplicada por cero, y la ecuación de proyección se reduce a las ecuaciones (9) y (10) del enunciado:

$$
\lambda\,m=HP,\qquad H=K\,[r_1\ \ r_2\ \ t]\in\mathbb{R}^{3\times3},
$$

con $P=(U,V,1)^\top$ y $m=(u,v,1)^\top$. Como $\lambda$ absorbe cualquier factor de escala, $H$ está definida salvo escala (9 entradas, 8 grados de libertad).

Igual que en el Ejercicio 1, se abre la ecuación fila por fila y se elimina $\lambda_i$ reemplazando la tercera igualdad en las otras dos:

$$
(h_1^\top-u_ih_3^\top)P_i=0,\qquad (h_2^\top-v_ih_3^\top)P_i=0,
$$

donde $h_j^\top$ es la fila $j$ de $H$. Son ecuaciones lineales en $h=(h_1;h_2;h_3)\in\mathbb{R}^9$ (las filas de $H$ una detrás de otra). Cada punto aporta dos filas:

$$
Q_i=
\begin{bmatrix}
U_i&V_i&1&0&0&0&-u_iU_i&-u_iV_i&-u_i\\
0&0&0&U_i&V_i&1&-v_iU_i&-v_iV_i&-v_i
\end{bmatrix}.
$$

Apilando las $n$ correspondencias (ecuación 11 del enunciado):

$$
\boxed{Qh=0},\qquad \boxed{Q\in\mathbb{R}^{2n\times9}},\qquad Q\in\mathbb{R}^{154\times9}\ \text{en nuestro caso}.
$$

---

## 3. Rango de $Q$ y cantidad de puntos

Queremos que la solución sea única salvo escala, es decir, $\dim(\ker(Q))=1$. Por el teorema de la dimensión (Teorema 1.1), como $Q$ tiene 9 columnas:

$$
9=\dim(\ker(Q))+\operatorname{rango}(Q)\quad\Longrightarrow\quad \boxed{\operatorname{rango}(Q)=8}.
$$

Como $\operatorname{im}(Q)\subseteq\mathbb{R}^{2n}$, se cumple $\operatorname{rango}(Q)\leq2n$, y para que el rango sea 8 hace falta $2n\geq8$:

$$
\boxed{n\geq4}.
$$

---