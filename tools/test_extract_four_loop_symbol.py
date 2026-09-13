"""Parser, partitioning and symmetry checks; no training or network."""

from fractions import Fraction
import unittest

from extract_four_loop_symbol import Parser, PartitionedExtractor, check_symmetry
from verify_four_loop_export import forward_verify


class FourLoopTests(unittest.TestCase):
    def test_power_precedence(self):
        self.assertEqual(Parser("-2^2 + (-2)^2 + 3^0").parse(), {(): Fraction(1)})

    def test_mzv_and_zeta(self):
        self.assertEqual(Parser("24*(MZV[{5,3}]-Zeta[2]*Zeta[3]^2)").parse(), {
            (("MZV", 5, 3),): Fraction(24),
            (("Zeta", 2), ("Zeta", 3), ("Zeta", 3)): Fraction(-24)})

    def test_scalar_arithmetic(self):
        self.assertEqual(Parser("(2*g[1]-g[2])*YE[7,3]/32").parse(), {
            (("YE", 7, 3), ("g", 1)): Fraction(1, 16),
            (("YE", 7, 3), ("g", 2)): Fraction(-1, 32)})

    def test_reject_syntax(self):
        for expression in ('Get["x"]', 'Run["x"]', 'ToExpression["1"]', '1;2',
                           '2^-1', '2^5', '2^(1+1)', 'MZV[{3,5}]', 'MZV[{5,3,1}]',
                           '1/g[1]', '1.5', '1/0', 'YE[8,0]', '1 2',
                           '__import__("os")', '(' * 65 + '1' + ')' * 65):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                Parser(expression).parse()

    def test_larger_polynomial(self):
        expression = '+'.join(f'g[{i}]*YE[7,{j}]' for i in range(1, 66) for j in range(1, 34))
        self.assertEqual(len(Parser(expression).parse()), 2145)

    def test_partition_order_cancellation_constant(self):
        a, b, c = ('YE', 1, 1), ('YE', 2, 1), ('YE', 2, 2)
        d, e, z = ('YE', 3, 1), ('YE', 3, 2), ('YE', 3, 3)
        graph = {a: {0: {None: -1}}, b: {3: {a: 2}}, c: {4: {a: 3}},
                 d: {5: {b: 1}, 3: {c: 2}}, e: {5: {b: -1}}, z: {}}
        root = {d: Fraction(1, 4), e: Fraction(1, 4), z: Fraction(7)}
        expected = {bytes([0, 4, 3]): Fraction(-3, 2)}
        for cutoff in (1, 2, 3, 4):
            self.assertEqual(PartitionedExtractor(graph, cutoff).amplitude(root), expected)

    def test_cutoff_bound(self):
        with self.assertRaises(ValueError):
            PartitionedExtractor({}, cutoff=8)

    def test_symmetry(self):
        symbol = {bytes([i, i + 3]): Fraction(1) for i in range(3)}
        check_symmetry(symbol)
        symbol[bytes([0, 3])] = Fraction(2)
        with self.assertRaises(ValueError):
            check_symmetry(symbol)

    def test_forward_verifier_exact_support(self):
        a, b, c = ('YE', 1, 1), ('YE', 2, 1), ('YE', 2, 2)
        d, e = ('YE', 3, 1), ('YE', 3, 2)
        graph = {a: {0: {None: -1}}, b: {3: {a: 2}}, c: {4: {a: 3}},
                 d: {5: {b: 1}, 3: {c: 2}}, e: {5: {b: -1}}}
        root = {d: Fraction(1, 4), e: Fraction(1, 4)}
        expected = {bytes([0, 4, 3]): Fraction(-3, 2)}
        report = forward_verify(graph, root, expected)
        self.assertEqual(report['all_nonzero_rows_verified_by_forward_propagation'], 1)
        for bad in ({}, {bytes([0, 4, 3]): Fraction(3, 2)},
                    {**expected, bytes([0, 3, 5]): Fraction(1)}):
            with self.assertRaises(ValueError):
                forward_verify(graph, root, bad)


if __name__ == '__main__':
    unittest.main()
