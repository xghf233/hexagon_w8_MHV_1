"""Small parser/recursion checks; no network, Wolfram, data download or training."""

from fractions import Fraction
import unittest

from extract_three_loop_symbol import Parser, SymbolExtractor


class ParserTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(Parser("(2 + 3*4)/7 - 1").parse(), {(): Fraction(1)})

    def test_distribution(self):
        actual = Parser("(-g[2] + 2*g[3])*YE[5, 4]/16").parse()
        self.assertEqual(actual, {
            (("YE", 5, 4), ("g", 2)): Fraction(-1, 16),
            (("YE", 5, 4), ("g", 3)): Fraction(1, 8)})

    def test_zeta_base_point(self):
        self.assertEqual(Parser("(Zeta[6]*(2*g[92]))/2").parse(),
                         {(("Zeta", 6), ("g", 92)): Fraction(1)})

    def test_zero(self):
        self.assertEqual(Parser("g[1] - g[1]").parse(), {})

    def test_reject_unsafe_or_unsupported(self):
        examples = ('Get["a"]', 'Run["a"]', 'ToExpression["a"]',
                    '__import__("os")', 'Sin[1]', '1;2', 'g[1]^2',
                    '1/0', '1/(g[1])', '1.5', 'YE[6,0]', 'g[1] trailing',
                    '(* comment *) 1', '1 2', '(' * 65 + '1' + ')' * 65)
        for expression in examples:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    Parser(expression).parse()

    def test_symbol_order_and_constant(self):
        a = ("YE", 1, 1)
        b = ("YE", 2, 1)
        c = ("YE", 2, 2)
        extractor = SymbolExtractor({a: {0: {None: -1}}, b: {3: {a: 2}}, c: {}})
        self.assertEqual(extractor.basis_symbol(b), {bytes([0, 3]): -2})
        self.assertEqual(extractor.query_basis(b, bytes([0, 3])), -2)
        self.assertEqual(extractor.query_basis(b, bytes([3, 0])), 0)
        self.assertEqual(extractor.basis_symbol(c), {})
        self.assertEqual(extractor.amplitude({b: Fraction(1, 4)}), {bytes([0, 3]): Fraction(-1, 2)})


if __name__ == "__main__":
    unittest.main()
