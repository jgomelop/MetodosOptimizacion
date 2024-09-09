def first_derivative(f, h=1e-4):
    return lambda x: (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)

def second_derivative(f, h=1e-3):
  return lambda x: (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12 * h**2)