from .Base import IBaseMethod, IOptimizationMethod
from .utils.Errors import relative_error as get_relative_error
from .utils.NumericalDifferentiation import first_derivative, second_derivative
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

class _OptimizationMethodBase(IBaseMethod, IOptimizationMethod):
    def __init__(self, f, opt, error, table):
        self.function = f
        self.estimate_opt = opt
        self.relative_error = error
        self.table = table
        print(
            f"Estimated opt: {self.estimate_opt}\n"
            f"Estimated relative error: {self.relative_error}"
            )

    def plot_function(self, lims = (-10,10)):
        assert lims[0] < 0, "Límite inferior debe ser menor que cero"
        assert lims[1] > 0, "Límite superior debe ser mayor que cero"
        opt = self.estimate_opt
        f = self.function
        x = np.linspace(opt + lims[0], opt + lims[1], 100)
        fopt = f(opt)
        fig, ax = plt.subplots()
        ax.scatter(
            x=[opt], 
            y=[fopt], 
            color='red', 
            s = 5, 
            label = f"Óptimo en x = {opt}",
            zorder = 4)
        ax.plot(x, f(x), zorder = 3)
        ax.axhline(y=fopt, color='black', linestyle=':', linewidth=1 , zorder = 2)
        ax.axvline(x=opt, color='black', linestyle=':', linewidth=1, zorder = 1)
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

    def get_estimated_opt(self):
        return self.estimate_opt

    def get_relative_error(self):
        return self.relative_error
    
    def get_table(self):
        return self.table
        
    def print_table(self):
        print(self.table) 

class Newton(_OptimizationMethodBase):
    def __init__(self, 
                x0, 
                f, 
                df = None, 
                ddf= None, 
                tol= 1e-3, 
                max_iter= 100):

        est_opt, rel_error, table_rows = self._job(x0, f, df, ddf, tol, max_iter)
        
        # Construcción de la tabla informativa.
        # Siempre poner "i" como primer elemento en la lista de columnas
        # siempre colocar el error como última columna
        COLUMNS = ["i", "xi", "f(x)","f'(x)", "f''(x)", "Relative_error"]
        table = pd.DataFrame(data = table_rows, columns = COLUMNS)
        table = table.set_index("i")
        super().__init__(f, est_opt, rel_error, table)

    def _job(self, x0, f, df, ddf, tol = 1e-5, max_iter = 100):
        table_rows = [] # filas de la tabla informativa
        ROUNDING_VALUE = int(abs(np.log10(tol)))

        df = df if df else first_derivative(f)
        ddf = ddf if ddf else second_derivative(f)
        xi = x0
        for i in range(max_iter):
            fx = f(xi)
            dfx = df(xi)
            ddfx = ddf(xi)

            if ddfx == 0:
                print(f"f'' = 0 en iteración {i}")
                break

            x_next = xi - dfx/ ddfx
            if x_next == 0:
                print(f"xi = 0 en iteración {i}")
                break
            ea = get_relative_error(x_next,xi)
            table_rows.append([i, x_next, fx, dfx, ddfx,ea])
            # Convergencia por tolerancia

            if ea < tol:
                break

            xi = x_next

        x_next = np.round(x_next, ROUNDING_VALUE)
        ea = np.round(ea, ROUNDING_VALUE)
        table_rows = np.round(table_rows, ROUNDING_VALUE)
        return x_next, ea, table_rows

class ParabolicInterpolation(_OptimizationMethodBase):
    def __init__(self, f, x0, x1, x2, tol = 1e-3, max_iter = 50):
        est_opt, rel_error, table_rows = self._job(f, x0, x1, x2, tol, max_iter)

        # Construcción de la tabla informativa.
        # Siempre poner "i" como primer elemento en la lista de columnas
        # siempre colocar el error como última columna
        COLUMNS = [
            "i", 
            "x0", "f(x0)",
            "x1", "f(x1)",
            "x2", "f(x2)",
            "x3", "f(x3)",
            "Relative_error"]
        table = pd.DataFrame(data = table_rows, columns = COLUMNS)
        table = table.set_index("i")
        super().__init__(f, est_opt, rel_error, table)

    def _x_new(f, x0,x1,x2):
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)
        num = fx0*(x1**2-x2**2)+fx1*(x2**2-x0**2)+fx2*(x0**2-x1**2)
        den = 2.*fx0*(x1-x2)+2.*fx1*(x2-x0)+2.*fx2*(x0-x1)
        return num/den, fx0, fx1, fx2

    def _job(self, f, x0, x1,x2, tol, max_iter):
        table_rows = []
        ROUNDING_VALUE = int(abs(np.log10(tol)))

        for i in range(max_iter):
            fx0 = f(x0)
            fx1 = f(x1)
            fx2 = f(x2)
            num = fx0*(x1**2-x2**2)+fx1*(x2**2-x0**2)+fx2*(x0**2-x1**2)
            den = 2.*fx0*(x1-x2)+2.*fx1*(x2-x0)+2.*fx2*(x0-x1)
            if den==0: 
                print(f"Denominador cero en iteracion {i}") 
                break
            x3 = num/den
            fx3 = f(x3)
            ea = get_relative_error(x3, x1)
            table_rows.append([
                i,
                x0, fx0, 
                x1, fx1, 
                x2, fx2, 
                x3, fx3, 
                ea])
            if ea < tol: break
            if x3 > x1: x0 = x1   
            else: x2 = x1
            x1 = x3

        x3 = np.round(x3, ROUNDING_VALUE)
        ea = np.round(ea, ROUNDING_VALUE)
        table_rows = np.round(table_rows, ROUNDING_VALUE)
        return x3, ea, table_rows


class GoldenSectionSearch(_OptimizationMethodBase):
    def __init__(self, f, xlow, xhigh, minimize = False, tol = 1e-3, max_iter = 100):
        est_opt, rel_error, table_rows = self._job(f, xlow, xhigh, minimize, tol, max_iter)
        # Construcción de la tabla informativa.
        # Siempre poner "i" como primer elemento en la lista de columnas
        # siempre colocar el error como última columna
        COLUMNS = [
            "i", 
            "x1", "f(x1)",
            "x2", "f(x2)",
            "Relative_error"]
        table = pd.DataFrame(data = table_rows, columns = COLUMNS)
        table = table.set_index("i")
        super().__init__(f, est_opt, rel_error, table)

    def _job(self, f, xlow, xhigh, minimize, tol, max_iter):
        cond = lambda x,y: (x < y) if minimize else (x > y) # Maximizar es el default.
        table_rows = []
        ROUNDING_VALUE = int(abs(np.log10(tol)))

        R = (5**0.5 - 1) / 2 
        xl = xlow 
        xu = xhigh
        d = R*(xu - xl)
        x1 = xl + d
        x2 = xu - d
        f1 = f(x1)
        f2 = f(x2)
        if cond(f1, f2):
            xopt = x1 
            fx = f1 
        else:
            xopt = x2 
            fx = f2 
        for i in range(1, max_iter+1):
            d = R*d
            xint = xu - xl
            if cond(f1, f2):
                xl = x2 
                x2 = x1 
                x1 = xl + d
                f2 = f1 
                f1 = f(x1)
            else:
                xu = x1 
                x1 = x2 
                x2 = xu - d
                f1 = f2 
                f2 = f(x2)
            if cond(f1, f2):
                xopt = x1 
                fx = f1 
            else:
                xopt = x2 
                fx = f2 
            try:
                # El cálculo del error cambia un poco, de acuerdo al libro
                ea = (1. - R)*abs(xint/xopt)
                
                table_rows.append([
                    i,
                    x1, f1, 
                    x2, f2, 
                    ea
                ])
            except:
                break
            if ea <= tol:
                break

        xopt = np.round(xopt, ROUNDING_VALUE)
        ea = np.round(ea, ROUNDING_VALUE)
        table_rows = np.round(table_rows, ROUNDING_VALUE)
        return xopt, ea, table_rows

class RandomSearch():
    def __init__(self, f, x_bounds = (0,1), y_bounds = (0,1), minimize = False,  max_iters = [50000], 
                cifras_decimales = 4):  
        
        if isinstance(max_iters,int): max_iters = [max_iters]
        elif isinstance(max_iters,list):
            if not all(isinstance(sub, int) for sub in max_iters):
                raise TypeError("Elementos de la lista max_iters deben ser int.")
        else:
            raise TypeError("max_iters debe int o lista de int.")

        table = []
        for i, max_iter in enumerate(max_iters):
            max_x, max_y, max_fxy = self._job(f, x_bounds, y_bounds, minimize, max_iter, cifras_decimales)
            table.append([i, max_iter, max_x, max_y, max_fxy])
        COLUMNS = ["i", "max_iter", "x","y", "f(x,y)"]
        table = pd.DataFrame(data = table, columns = COLUMNS)
        self.table = table.set_index("i")
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.f = f 
    
    def _random_sample(self, lower_bound, upper_bound):
        return lower_bound + (upper_bound - lower_bound)*np.random.uniform(0,1)

    def _job(self, f, x_bounds, y_bounds, minimize, max_iter, cifras_decimales):
        cond = lambda a,b: (a < b) if minimize else (a > b) # Maximizar es el default.
        x_lower, x_upper = x_bounds
        y_lower, y_upper = y_bounds

        # Inicialziació naleatoria de de max_x y max_y
        max_x   = np.round(self._random_sample(x_lower, x_upper), cifras_decimales)
        max_y   = np.round(self._random_sample(x_lower, x_upper), cifras_decimales)
        max_fxy = np.round(f(max_x, max_y) , cifras_decimales)

        for _ in range(max_iter-1):
            xr = np.round(self._random_sample(x_lower, x_upper), cifras_decimales)
            yr = np.round(self._random_sample(y_lower, y_upper), cifras_decimales)
            fxy = np.round(f(xr, yr), cifras_decimales)
            if cond(fxy, max_fxy):
                max_fxy = fxy 
                max_x = xr 
                max_y = yr
        return max_x, max_y, max_fxy

    def get_table(self):
        return self.table   

    def print_table(self):
        print(self.table)   
    
    def plot(self, delta = 0.025, niveles = 10, fontsize = 10):
        x_lower, x_upper = self.x_bounds
        y_lower, y_upper = self.y_bounds
        x = np.arange(x_lower, x_upper, delta)
        y = np.arange(y_lower, y_upper, delta)
        X, Y = np.meshgrid(x, y)
        Z = self.f(X,Y)

        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z, levels = niveles)
        ax.clabel(CS, inline=True, fontsize = fontsize)
        # Graficamos los valores óptimos
        ax.scatter(
            x=self.table.x.to_numpy(), 
            y=self.table.y.to_numpy(), 
            color='red', 
            s = 5)
        ax.set_title('Gráfico valores óptimos de la función')
        plt.show()

