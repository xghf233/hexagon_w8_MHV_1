from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import extract_five_loop_symbol as five


class FiveLoopTests(unittest.TestCase):
    def test_large_parser(self):
        text = '+'.join(f'g[{i}]*YE[9,{j}]' for i in range(1,100) for j in range(1,100))
        self.assertEqual(len(five.Parser(text).parse()),9801)

    def test_constants_and_precedence(self):
        self.assertEqual(five.Parser('-2^2+(-2)^2+3^0').parse(),{():Fraction(1)})
        self.assertEqual(five.Parser('MZV[{7,3}]-Zeta[4]*Zeta[3]^2').parse(),
                         {(("MZV",7,3),):Fraction(1),(("Zeta",3),("Zeta",3),("Zeta",4)):Fraction(-1)})

    def test_reject(self):
        for text in ('Get["x"]','Run["x"]','1;2','MZV[{9,3}]','2^5','2^-1','1/g[1]',
                     '1.0','1 2','1/0','YE[10,0]','('*65+'1'+')'*65):
            with self.subTest(text=text), self.assertRaises(ValueError):
                five.Parser(text).parse()

    def test_word_code(self):
        for word in (bytes(10),bytes([8]*10),bytes([0,1,2,3,4,5,6,7,8,0])):
            self.assertEqual(five.decode_word(five.word_id(word)),word)

    def test_rational_transition_clearing(self):
        a,b,c=('YE',1,1),('YE',2,1),('YE',3,1)
        graph={a:{0:{None:Fraction(1)}},b:{3:{a:Fraction(1,2)}},c:{4:{b:Fraction(1,3)}}}
        scales=five.integerize(graph)
        self.assertEqual(scales['cumulative_symbol_scale'][3],6)
        state,denominator=five.scaled_root({c:Fraction(1,4)},6)
        actual=dict(five.Backward(graph,five.Budget(30)).records(state,3))
        self.assertEqual({code:Fraction(n,denominator) for code,n in actual.items()},
                         {five.word_id([0,3,4]):Fraction(1,24)})

    def test_bidirectional(self):
        a,b,c,d,e=('YE',1,1),('YE',2,1),('YE',2,2),('YE',3,1),('YE',3,2)
        graph={a:{0:{None:-1}},b:{3:{a:2}},c:{4:{a:3}},d:{5:{b:1},3:{c:2}},e:{5:{b:-1}}}
        root={d:Fraction(1,4),e:Fraction(1,4)}
        state,denominator=five.scaled_root(root)
        budget=five.Budget(30)
        left=dict(five.Backward(graph,budget).records(state,3))
        right=dict(five.Forward(graph,state,budget).records())
        self.assertEqual(left,right)
        self.assertEqual(left,{five.word_id([0,4,3]):-6})
        self.assertEqual(denominator,4)

    def test_sqlite_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            conn=five.connect_new(Path(directory)/'test.sqlite')
            five.insert_records(conn,[(1,2)],five.Budget(30))
            with self.assertRaises(five.sqlite3.IntegrityError):
                five.insert_records(conn,[(1,3)],five.Budget(30))
            conn.close()

    def test_native_verifier_rejects_wrong_coefficient(self):
        a,b,c=('YE',1,1),('YE',2,1),('YE',3,1)
        graph={a:{0:{None:-1}},b:{3:{a:2}},c:{4:{b:3}}}
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            budget=five.Budget(30)
            native=five.native_setup(graph,{c:1},budget,directory)
            for name,value in [('good',-6),('bad',6)]:
                path=directory/(name+'.bin')
                path.write_bytes(b'HX5B001\n'+five.struct.pack('<QIq',1,five.word_id([0,3,4]),value))
                if name=='good':
                    self.assertEqual(five.native_run(native,budget,reference=path)['rows'],1)
                else:
                    with self.assertRaises(five.subprocess.CalledProcessError):
                        five.native_run(native,budget,reference=path)

    def test_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            conn=five.connect_new(directory/'data.sqlite')
            word=bytes([0,0,0,0,0,0,0,0,0,3])
            five.insert_records(conn,[(five.word_id(word),-96)],five.Budget(30))
            stats,_=five.export_data(conn,directory/'data.jsonl.gz',64,five.Budget(30))
            self.assertEqual(stats['rows'],1)
            self.assertEqual(stats['coefficient_histogram'],{'-3/2':1})
            conn.close()


if __name__=='__main__':
    unittest.main()
