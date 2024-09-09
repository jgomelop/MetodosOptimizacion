from .Base import IBaseMethod, IRootFindingMethod
from .utils.Errors import relative_error as get_relative_error
from .utils.NumericalDifferentiation import first_derivative
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

class _RootFindingMethodBase(IBaseMethod, IRootFindingMethod):
    def __init__(self, f, root, error, table):
        self.function = f
        self.estimated_root = root
        self.relative_error = error
        self.table = table
        print(
            f"Estimated root: {self.estimated_root}\n"
            f"Estimated relative error: {self.relative_error}"
            )

    def plot_function(self, lims = (-10,10)):
        assert lims[0] < 0, "Límite inferior debe ser menor que cero"
        assert lims[1] > 0, "Límite superior debe ser mayor que cero"
        root = self.estimated_root
        f = self.function
        x = np.linspace(root + lims[0], root + lims[1], 100)
        fig, ax = plt.subplots()
        ax.scatter(
            x=[root], 
            y=[0.0], 
            color='red', 
            s = 5, 
            label = f"Raíz en x = {root}",
            zorder = 4)
        ax.plot(x, f(x), zorder = 3)
        ax.axhline(y=0.0, color='black', linestyle=':', linewidth=1 , zorder = 2)
        ax.axvline(x=root, color='black', linestyle=':', linewidth=1, zorder = 1)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Gráfico de la función')
        ax.legend()
        plt.show()
        plt.close()

    def plot_error(self):
        # Plot the function
        # y debe tener el mismo nombre que el dado en el constructor
        y = self.table.columns.to_numpy()[-1]
        self.table.plot(y=y, title='Error relativo')
        plt.xlabel('Iteración')
        plt.ylabel('Error')
        plt.show()
        plt.close()

    def get_estimated_root(self):
        return self.estimated_root

    def get_relative_error(self):
        return self.relative_error
    
    def get_table(self):
        return self.table
        
    def print_table(self):
        print(self.table) 


class FalsePosition(_RootFindingMethodBase):
    def __init__(self, f, xl, xu, tol = 1e-3, maxiter=50):
        est_root, rel_error, table_rows = self._job(f, xl, xu, tol, maxiter)
        
        # Construcción de la tabla informativa.
        # Siempre poner "i" como primer elemento en la lista de columnas
        # siempre colocar el error como última columna
        COLUMNS = ["i", "X_Lower", "X_Upper","X_Root", "Relative_error"]
        table = pd.DataFrame(data = table_rows, columns = COLUMNS)
        table = table.set_index("i")
        super().__init__(f, est_root, rel_error, table)

    def _job(self, f, xl, xu, tol, maxiter):
        # Tabla informativa
        # iteración, xr, Error relativo
        table_rows = []
        ROUNDING_VALUE = int(abs(np.log10(tol)))

        # Test f(xl)*f(xu) < 0
        fl = f(xl)
        fu = f(xu)
        test = fl*fu
        if test >= 0:
            raise Exception("Rango xl a xu inválido. f_xl*f_xu >= 0")

        xr_old = xu # Thrash value
        xr = (fu*(xl - xu))/(fl-fu) # False position
        for i in range(maxiter):
            fr = f(xr)
            test = fl*fr

            if test < 0: 
                xu = xr # subintervalo inferior
                ea = get_relative_error(x=xr, y=xr_old)
            elif test > 0: 
                xl = xr # Subintervalo superior
                fl = fr
                ea = get_relative_error(x=xr, y=xr_old)
            else:
                ea = 0
            table_rows.append([i, xl, xu, xr, ea])
            if ea < tol:
                break

            xr_old = xr
            xr = xu - (fu*(xl - xu))/(fl-fu) # Formula Falsa pos.

        # Redondeando resultados hasta exp de tol.
        xr = np.round(xr, ROUNDING_VALUE)
        ea = np.round(ea, ROUNDING_VALUE)
        table_rows = np.round(table_rows, ROUNDING_VALUE)
        return xr, ea, table_rows

class Bisection(_RootFindingMethodBase):
    def __init__(self, f, xl, xu, tol = 1e-3, maxiter=50):
        est_root, rel_error, table_rows = self._job(f, xl, xu, tol, maxiter)
        
        # Construcción de la tabla informativa.
        # Siempre poner "i" como primer elemento en la lista de columnas
        # siempre colocar el error como última columna
        COLUMNS = ["i", "X_Lower", "X_Upper","X_Root", "Relative_error"]
        table = pd.DataFrame(data = table_rows, columns = COLUMNS)
        table = table.set_index("i")
        super().__init__(f, est_root, rel_error, table)

    # El método en cuestión
    def _job(self, f, xl, xu, tol, maxiter):
        # Tendra listas de 6 elementos
        table_rows = [] # filas de la tabla informativa
        ROUNDING_VALUE = int(abs(np.log10(tol)))

        # Test f(xl)*f(xu) < 0
        fl = f(xl)
        fu = f(xu)
        test = fl*fu
        if test >= 0:
            raise Exception("Rango xl a xu inválido. f_xl*f_xu >= 0")
        xr_old = xu # Thrash value
        xr = (xl + xu)/2
        for i in range(maxiter):
            fr = f(xr)
            test = fl*fr

            if test < 0: 
                xu = xr # subintervalo inferior
                ea = get_relative_error(x=xr, y=xr_old)
            elif test > 0: 
                xl = xr # Subintervalo superior
                fl = fr
                ea = get_relative_error(x=xr, y=xr_old)
            else:
                ea = 0

            table_rows.append([i, xl, xu, xr, ea])
            if ea < tol:
                break

            xr_old = xr
            xr = (xl + xu)/2

        xr = np.round(xr, ROUNDING_VALUE)
        ea = np.round(ea, ROUNDING_VALUE)
        table_rows = np.round(table_rows, ROUNDING_VALUE)
        return xr, ea, table_rows


class NewtonRaphson(_RootFindingMethodBase):
    def __init__(self, x0, f, df = None, tol = 1e-3, max_iter = 100):
        est_root, rel_error, table_rows = self._job(x0,f,df,tol,max_iter)
        
        # Construcción de la tabla informativa.
        # Siempre poner "i" como primer elemento en la lista de columnas
        # siempre colocar el error como última columna
        COLUMNS = ["i", "x_i", "Relative_error"]
        table = pd.DataFrame(data = table_rows, columns = COLUMNS)
        table = table.set_index("i")
        super().__init__(f, est_root, rel_error, table)
        
    def _job(self,x0, f, df,tol, max_iter):
        table_rows = [] # filas de la tabla informativa
        ROUNDING_VALUE = int(abs(np.log10(tol)))

        # Si se pasa la derivada analítica usar, 
        # De lo contrario, usar la diferenciación numérica.
        df = df if df else first_derivative(f)
        xi = x0
        for i in range(max_iter):
            fx = f(xi)
            dfx = df(xi)
            if dfx == 0:
                raise ValueError("Derivada es cero => División por cero.")
                
            x_next = xi - fx / dfx
            if x_next == 0:
                print(f"xi = 0 en iteración {i}")
                break
            ea = abs(x_next - xi)
            table_rows.append([i, x_next, ea])
            if ea < tol:
                break
            xi = x_next

        x_next = np.round(x_next, ROUNDING_VALUE)
        ea = np.round(ea, ROUNDING_VALUE)
        table_rows = np.round(table_rows, ROUNDING_VALUE)
        return x_next, ea, table_rows


class FixedPoint(_RootFindingMethodBase):
    def __init__(self, x0, f, g, tol = 1e-3, max_iter = 100):
        est_root, rel_error, table_rows = self._job(g, x0, tol, max_iter)
        
        # Construcción de la tabla informativa.
        # Siempre poner "i" como primer elemento en la lista de columnas
        # siempre colocar el error como última columna
        COLUMNS = ["i", "x_i", "Relative_error"]
        table = pd.DataFrame(data = table_rows, columns = COLUMNS)
        table = table.set_index("i")
        super().__init__(f, est_root, rel_error, table)

    def _job(self, g, x0, tol, max_iter):
        table_rows = [] # filas de la tabla informativa
        ROUNDING_VALUE = int(abs(np.log10(tol)))

        xr = x0
        for i in range(max_iter):
            xrold = xr
            xr = g(xrold)
            if xr != 0:
                ea = get_relative_error(xr,xrold)
                table_rows.append([i, xr, ea])
            if ea < tol:
                break

        xr = np.round(xr, ROUNDING_VALUE)
        ea = np.round(ea, ROUNDING_VALUE)
        table_rows = np.round(table_rows, ROUNDING_VALUE)
        return xr, ea, table_rows