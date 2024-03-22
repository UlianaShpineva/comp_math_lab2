import numpy as np
import matplotlib.pyplot as plt


def root_exists(f, a0, b0):
    if f(a0) * f(b0) > 0:
        print("На введенном интервале отсутствуют корни уравнения или несколько корней")
        return False
    if not is_monotonic(f, a0, b0):
        print("На заданном интервале более одного корня")
        return False
    return True


def is_monotonic(func, a, b):
    x_values = np.linspace(a, b, num=1000)
    y_values = func(x_values)
    cnt = 0
    for i in range(1, len(x_values)):
        if y_values[i] * y_values[i - 1] < 0:
            cnt += 1
    if cnt > 1 or cnt == 0:
        return False
    elif cnt == 1:
        return True


def simple_graph(f, l, r):
    x = np.arange(l, r, 0.1)

    y = f(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


def sys_graph(f1, f2, l, r):
    y_values_1 = np.linspace(l, r, 400)
    x_values_1 = np.linspace(l, r, 400)

    z1 = np.zeros((len(x_values_1), len(y_values_1)))

    for i in range(len(x_values_1)):
        for j in range(len(y_values_1)):
            z1[i, j] = f1(y_values_1[j], x_values_1[i])

    y_values_2 = np.linspace(l, r, 400)
    x_values_2 = np.linspace(l, r, 400)

    z2 = np.zeros((len(x_values_2), len(y_values_2)))

    for i in range(len(x_values_2)):
        for j in range(len(y_values_2)):
            z2[i, j] = f2(y_values_2[j], x_values_2[i])

    plt.contour(y_values_1, x_values_1, z1, levels=[0], colors='r')
    plt.contour(y_values_2, x_values_2, z2, levels=[0], colors='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def chords(func, a0, b0, eps):
    x_pred = a0
    a = a0
    b = b0
    cnt = 0

    while True:
        x = (a * func(b) - b * func(a)) / (func(b) - func(a))

        if func(x) * func(b) < 0:
            a = x
        elif func(x) * func(a) < 0:
            b = x

        cnt += 1

        if abs(x - x_pred) <= eps and abs(func(x)) <= eps:
            return x, func(x), cnt

        x_pred = x


def newton(func, a0, b0, eps):
    if func(a0) * diff2(func, a0) > 0:
        x = a0
    elif func(b0) * diff2(func, b0) > 0:
        x = b0

    cnt = 0

    while True:
        x = x - func(x) / diff(func, x)
        cnt += 1

        if abs(func(x)) <= eps:
            return x, func(x), cnt


def simple_iteration(func, a0, b0, eps):
    x = a0
    cnt = 0
    if diff(func, (a0 + b0) / 2) > 0:
        la = -1
    else:
        la = 1
    la *= 1 / max(abs(diff(func, a0)), abs(diff(func, b0)))

    print("fi(a0) = ", diff_fi(la, a0, func), "\nfi(b0) = ", diff_fi(la, b0, func))
    fl = True
    cnt_fi = 0
    while True:
        if (abs(diff_fi(la, x, func)) > 1 or (diff_fi(la, a0, func)) > 1 or abs(diff_fi(la, b0, func)) > 1) and fl:
            print("Расходится")
            fl = False
        x = fi(func(x), la, x)
        cnt += 1
        cnt_fi +=1
        if abs(func(x)) <= eps or cnt_fi >= 1000:
            return x, func(x), cnt


def plot_function(f, a0, b0):
    x = np.linspace(a0 - 3, b0 + 3, 100)
    y = f(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.xlim(a0 - 3, b0 + 3)
    plt.ylim(min(y) - 3, max(y) + 3)
    plt.show()


def newton_sys(f1, f2, x0, y0, eps, jacobian):
    x, y = x0, y0
    x_pred, y_pred = x0, y0
    del_x = x - x_pred
    del_y = y - y_pred
    cnt = 0
    while True:
        jac = jacobian(x, y)
        f_val = np.array([f1(x, y), f2(x, y)])
        delta = np.linalg.solve(jac, -f_val)
        x, y = x + delta[0], y + delta[1]
        del_x = x - x_pred
        del_y = y - y_pred
        if abs(x - x_pred) <= eps and abs(y - y_pred) <= eps:
        # if abs(f1(x, y)) <= eps and abs(f2(x, y)) <= eps:
            break
        cnt += 1

        x_pred, y_pred = x, y
    print(f1(x, y), f2(x, y))
    return x, y, cnt, del_x, del_y


def plot_functions(f1, f2, x0, y0):
    y_values_1 = np.linspace(x0 - 3, x0 + 3, 400)
    x_values_1 = np.linspace(y0 - 3, y0 + 3, 400)

    z1 = np.zeros((len(x_values_1), len(y_values_1)))

    for i in range(len(x_values_1)):
        for j in range(len(y_values_1)):
            z1[i, j] = f1(y_values_1[j], x_values_1[i])

    x_values_2 = np.linspace(y0 - 3, y0 + 3, 400)
    y_values_2 = np.linspace(x0 - 3, x0 + 3, 400)

    z2 = np.zeros((len(x_values_2), len(y_values_2)))

    for i in range(len(x_values_2)):
        for j in range(len(y_values_2)):
            z2[i, j] = f2(y_values_2[j], x_values_2[i])

    plt.contour(y_values_1, x_values_1, z1, levels=[0], colors='r')
    plt.contour(y_values_2, x_values_2, z2, levels=[0], colors='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    # plt.axis('equal')
    plt.show()


def fi(fn, la, x):
    return x + la * fn


def diff(f, x0):
    h = 1e-6
    return (f(x0 + h) - f(x0)) / h


def diff2(f, x0):
    h = 1e-6
    return (f(x0 + h) - 2*f(x0) + f(x0 - h))/(pow(h, 2))


def diff_fi(la, x, f):
    h = 1e-6
    return (x + h + la * f(x + h) - x - la * f(x)) / h


def jacobian1(x, y):
    return np.array([[2 * x, 2 * y], [6 * x, -1]])


def jacobian2(x, y):
    return np.array([[-1, np.cos(y + 2)], [-np.sin(x - 2), 1]])


def f1_1(x, y):
    return x**2 + y**2 - 4


def f1_2(x, y):
    return 3 * x**2 - y


def f2_1(x, y):
    return np.sin(y + 2) - x - 1.5


def f2_2(x, y):
    return y + np.cos(x - 2) - 0.5


print("Выберите ввод данных:")
print("1. Консоль")
print("2. Файл")
input_type = int(input())

print("Выберите метод решения:")
print("1. Метод хорд")
print("2. Метод Ньютона")
print("3. Метод простой итерации")
print("4. Метод Ньютона для системы")
method = int(input())

if method != 4:
    print("Выберите уравнение для решения:")
    print("1. x^3 - x + 4 = 0")
    print("2. sin(x) = 0")
    print("3. x^3 + 2.64x^2 - 5.41x - 11.76")
    equation = int(input())

    if input_type == 1:
        if equation == 1:
            func = lambda x: x**3 - x + 4
            left = -5
            right = 5
        elif equation == 2:
            func = lambda x: np.sin(x)
            left = -5
            right = 5
        elif equation == 3:
            func = lambda x: x**3 + 2.64 * x**2 - 5.41 * x - 11.76
            left = -8
            right = 8

        simple_graph(func, left, right)

        a0 = float(input("Введите левую границу интервала: "))
        b0 = float(input("Введите правую границу интервала: "))
        eps = float(input("Введите точность: "))

        if not root_exists(func, a0, b0):
            exit()

        if method == 1:
            x, y, iteration = chords(func, a0, b0, eps)
        elif method == 2:
            x, y, iteration = newton(func, a0, b0, eps)
        elif method == 3:
            x, y, iteration = simple_iteration(func, a0, b0, eps)

        if x is not None:
            print(f"Корень уравнения: x = {x:.10f}")
            print(f"Значение функции в корне: f(x) = {y:.10f}")
            print(f"Число итераций: {iteration}")
            plot_function(func, a0, b0)

    elif input_type == 2:
        filename = input("Введите имя файла: ")
        with open(filename, "r") as file:
            a0 = float(file.readline().strip())
            b0 = float(file.readline().strip())
            eps = float(file.readline().strip())


        if equation == 1:
            func = lambda x: x**3 - x + 4
        elif equation == 2:
            func = lambda x: np.sin(x)
        elif equation == 3:
            func = lambda x: x**3 + 2.64 * x**2 - 5.41 * x - 11.76

        if not root_exists(func, a0, b0):
            exit()

        if method == 1:
            x, y, iteration = chords(func, a0, b0, eps)
        elif method == 2:
            x, y, iteration = newton(func, a0, b0, eps)
        elif method == 3:
            x, y, iteration = simple_iteration(func, a0, b0, eps)

        output_filename = input("Введите имя файла для вывода: ")
        with open(output_filename, "w") as file:
            if x is not None:
                file.write(f"Корень уравнения: x = {x:.6f}\n")
                file.write(f"Значение функции в корне: y = {y:.6f}\n")
                file.write(f"Число итераций: {iteration}")
            else:
                file.write("На введенном интервале более одного корня.")

        print(f"Результаты успешно записаны в файл {output_filename}")
elif method == 4:
    print("Выберите систему уравнений:")
    print("1. x^2 + y^2 - 4 = 0\n   x^2 - y - 1 = 0")
    print("2. sin(y + 2) - x - 1.5 = 0\n   y + cos(x - 2) - 0.5 = 0")
    sys = int(input())

    if sys == 1:
        f1, f2 = f1_1, f1_2
        jacobian = jacobian1
        left = -3
        right = 3
    elif sys == 2:
        f1, f2 = f2_1, f2_2
        jacobian = jacobian2
        left = -3
        right = 2

    sys_graph(f1, f2, left, right)

    x0 = float(input("Введите x0: "))
    y0 = float(input("Введите y0: "))
    eps = float(input("Введите точность: "))

    x, y, iter_count, d_x, d_y = newton_sys(f1, f2, x0, y0, eps, jacobian)

    print(f"Вектор неизвестных: x1 = {x}, x2 = {y}")
    print(f"Количество итераций: {iter_count}")
    print(f"Вектор погрешностей: [{d_x}, {d_y}]")

    plot_functions(f1, f2, x0, y0)