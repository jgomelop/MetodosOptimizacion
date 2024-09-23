# Metodos Optimizacion
Repositorio con métodos de búsqueda de raíces y máximos y mínimos para el curso de optimización de la Universidad de Antioquia 2024-2

## Guía de usuario
El proyecto se estructura de la siguiente manera: 

```
methods/
|___utils/
    |___Errors.py
    |___NumericalDifferentiation.py
|__Base.py
|___Optimize.py
|___RootFinding.py
```

La carpeta donde se encuentran los modulos es ```methods/```. Los dos módulos principales son ```Optimize.py``` y ```RootFinding.py```, pues estos contienen las clases que implementan cada método.

```Optimize.py``` contiene los métodos para buscar máximos y mínimos de funciones:
- ```Newton```
- ```ParabolicInterpolation```
- ```GoldenSectionSearch```
- ```RandomSearch```

```RootFinding.py``` implementa los métodos de búsquedas de raíces:
- ```FalsePosition ```
- ```Bisection```
- ```NewtonRaphson``` 
- ```FixedPoint```

Las clases ```_OptimizationMethodBase``` y ```_RootFindingMethodBase``` no deben ser directamente instanciadas, pues estas clases se usan como plantillas base para las clases mostradas anteriormente.

```Base.py``` contiene las interfaces que las clases implementan, exceptuando ```RandomSearch```. Por lo tanto, todas las clases implementarán los siguientes métodos de  la interfaz ```IBaseMethod```:
- ```plot_function```: para graficar la función
- ```plot_error```: para graficar el error por iteración
- ```get_relative_error```: para obtener el error relativo estimado final

```IRootFindingMethod``` es la interfaz para los métodos que buscan raíces, y contiene el método abstracto ```get_estimated_root```, para obtener el valor de $x$ estimado donde se aproxima a una raíz de la función. 

Similarmente, ```IOptimizationMethod``` es la interfaz para los métodos de optimización, requiriendo la implementación del método ```get_estimated_opt``` para obtener el valor de $x$ donde se encuentra un máximo o mínimo del a función.

La carpeta ```utils``` contiene algunas dependencias usadas por múltiples métodos, tanto de optimización como de búsqueda de raíces. 

### Ejemplos de uso

Lo primero que se debe hacer es tener la carpeta  ```methods/``` junto en el directorio raíz del proyecto donde serán importados los módulos.

```python
from methods.RootFinding import Bisection
import numpy as np

f = lambda x: np.exp(-x) - x
rf = Bisection(f=f,xl=-3, xu=2, tol=1e-3)
# Grafica la función indicando los puntos óptimos
rf.plot_function() 
# Imprime la tabla resumen 
print(rf.table)
# Grafica el error
rf.plot_error()
# Obtiene la raíz estimada
rf.get_estimated_root()
```

```python
from methods.Optimize import Newton
import numpy as np

f = lambda x: 2*np.sin(x) - x*x/10
opt = Newton(x0=2, f, tol=1e-6)

# Grafica la función indicando los puntos óptimos
opt.plot_function() 
# Imprime la tabla resumen 
print(opt.table)
# Grafica el error
opt.plot_error()
# Obtiene el valor de x donde la función se maximiza/minimiza
opt.get_estimated_opt()
```


### Random Search
Como se mencionó anteriormente, ```RandomSearch``` no implementa estas interfaces debido a su naturaleza. Los métodos principales de esta clase son:
- ```plot```: Grafica de la función con uno o varios puntos que aproximan el máximo o mínimo de la función $f(x,y)$.
- ```get_table```: Obtiene la tabla con el número máximo de iteraciones, $x$, $y$ y $f(x,y)$. Esta tabla puede tener varias filas como se explica más adelante.

```RandomSearch``` recibe los intervalos para $x$ y para $y$ como tuplas (ver ejemplo más abajo). Primero van los de $x$ y luego los de $y$.

```RandomSearch``` tiene como parámetro de entrada ```minimize```, que por defecto es ```False```. Establecer este parámetro como ```True``` hará que el algoritmo busque por mínimos de funciones. 

Por otro lado, tiene el parámetro ```max_iters```, que recibe un entero o una lista de enteros, indicando los diferentes valores para el máximo de iteraciones. Entonces, si se ingresan, por ejemplo, 4 valores dentro de esta lista, el algoritmo ejecutará 4 veces la búsqueda de los valores óptimos, de manera aleatoria, con un máximo de iteraciones según el i-ésimo valor de la lista definida por el usuario. Por lo tanto, la tabla resultante tendrá 4 filas, como se mencionó anteriormente.

Un ejemplo de uso es el siguiente:

```python
from methods.Optimize import RandomSearch

f = lambda x,y: y - x - 2*x**2 - 2*x*y - y**2
opt = RandomSearch(f, (-2,2), (1,3), False, max_iters= [1000, 2000])
opt.print_table()
opt.plot(delta = 0.0025)
```

Finalmente, ```RandomSearch``` también tiene como parámetro de entrada ```cifras_decimales```. Este parámetro sirve para controlar el redondeo de lso resultados. Esto se hace necesario por estabilidad numérica, pues algunas si se trabaja con una precisión muy alta o muy baja, pueden aparecer errores de redondeo, arrojando resultados extraños.

### GoldenSectionSearch

```GoldenSectionSearchmSearch``` tiene como parámetro de entrada ```minimize```, que por defecto es ```False```. Establecer este parámetro como ```True``` hará que el algoritmo busque por mínimos de funciones. 

## Google Colab
Para importar en colab:

```
!git clone https://github.com/jgomelop/MetodosOptimizacion.git
```
Posteriormente:

```python
from MetodosOptimizacion.methods.RootFinding import NewtonRaphson
```
## FAQ

**Import error por versiones de numpy**
Si aparece un ImportError, es por un problema con las versiones entre numpy y pandas. Se recomienda instalar una versión de numpy<2. En general, se recomienda no isntalar la última versión de python hasta la fecha, la 3.12, y trabajar con la versión 3.11.x

**Copio un código de ejemplo, y sólo aparece una gráfica**
Debes cerrar el primer gráfico para generar el segundo.












