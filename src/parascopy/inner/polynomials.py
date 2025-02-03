from collections import defaultdict
import itertools
import numpy as np
from numpy.polynomial import polynomial


class SparsePolynomial(defaultdict):
    def __init__(self):
        super().__init__(float)

    def __mul__(self, other):
        oth_items = list(other.items())
        res = SparsePolynomial()
        for a_power, a_coef in self.items():
            for b_power, b_coef in oth_items:
                res[a_power + b_power] += a_coef * b_coef
        return res

    def sqr(self):
        self_items = list(self.items())
        n = len(self_items)
        res = SparsePolynomial()
        for i, (a_power, a_coef) in enumerate(self_items):
            res[a_power * 2] += a_coef ** 2
            for j in range(i + 1, n):
                b_power, b_coef = self_items[j]
                res[a_power + b_power] += 2 * a_coef * b_coef
        return res

    def __getattr__(self, power):
        return self.get(power, 0)

    def __pow__(self, power):
        assert power >= 1
        x = self
        y = None
        while power > 1:
            if power % 2:
                y = x * y if y is not None else x
            x = x.sqr()
            power //= 2
        return x * y if y is not None else x

    def __str__(self):
        if not self:
            return '0'
        return ' + '.join(itertools.starmap('{1:.4f} x^{0}'.format, sorted(self.items())))

    def get_slice(self, start, out):
        for i in range(len(out)):
            out[i] = self.get(start + i, 0)


class DensePolynomial(np.ndarray):
    def __new__(cls, len):
        self = np.zeros(len).view(cls)
        return self

    def __mul__(self, other):
        return polynomial.polymul(self, other).view(self.__class__)

    def __pow__(self, power):
        return polynomial.polypow(self, power).view(self.__class__)

    def __str__(self):
        if not len(self):
            return '0'
        return ' + '.join('{:.4f} x^{}'.format(coef, i) for i, coef in enumerate(self) if coef)

    def get_slice(self, start, out):
        if start > len(self):
            return
        end = start + len(out)
        res = self[start : end]
        out[:len(res)] = res


def multiply_polynomials_f_values(alleles, powers):
    copy_num = len(alleles)
    n_alleles = max(alleles) + 1
    sum_power = sum(powers)

    max_power = 0
    f_powers = np.zeros(copy_num, dtype=np.int64)
    for i in range(copy_num - 1, -1, -1):
        f_powers[i] = max_power + 1
        max_power += (max_power + 1) * powers[i]
    f_modulo = max_power + 1
    x_powers = []
    for _ in powers:
        x_powers.append(max_power + 1)
        max_power += x_powers[-1] * sum_power

    res_poly = None
    if max_power >= 1000:
        create_polynomial = SparsePolynomial
    else:
        uni_len = x_powers[-1] + f_powers[0] - f_modulo + 1
        create_polynomial = lambda: DensePolynomial(uni_len)

    not_f_coef = 1 / (n_alleles - 1) if n_alleles > 1 else None
    for i, power in enumerate(powers):
        if not power:
            continue
        curr_f_power = f_powers[i]
        curr_allele = alleles[i]

        uni_var = create_polynomial()
        for allele, x_power in enumerate(x_powers):
            if allele == curr_allele:
                uni_var[curr_f_power + x_power - f_modulo] = 1
            else:
                uni_var[x_power - f_modulo] = not_f_coef
                uni_var[curr_f_power + x_power - f_modulo] = -not_f_coef
        if power > 1:
            uni_var = uni_var ** power
        res_poly = res_poly * uni_var if res_poly is not None else uni_var
    return f_powers, x_powers, res_poly
