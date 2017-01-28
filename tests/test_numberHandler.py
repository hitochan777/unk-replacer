from unittest import TestCase
from replacer.number_normalizer import NumberHandler


class TestNumberHandler(TestCase):
    def test_split_number_0_to_12(self):
        self.assertEqual(NumberHandler.process_number('０'), "<@num:0>")
        self.assertEqual(NumberHandler.process_number('１'), "<@num:1>")
        self.assertEqual(NumberHandler.process_number('２'), "<@num:2>")
        self.assertEqual(NumberHandler.process_number('３'), "<@num:3>")
        self.assertEqual(NumberHandler.process_number('４'), "<@num:4>")
        self.assertEqual(NumberHandler.process_number('５'), "<@num:5>")
        self.assertEqual(NumberHandler.process_number('６'), "<@num:6>")
        self.assertEqual(NumberHandler.process_number('７'), "<@num:7>")
        self.assertEqual(NumberHandler.process_number('８'), "<@num:8>")
        self.assertEqual(NumberHandler.process_number('９'), "<@num:9>")
        self.assertEqual(NumberHandler.process_number('１０'), "<@num:10>")
        self.assertEqual(NumberHandler.process_number('１１'), "<@num:11>")
        self.assertEqual(NumberHandler.process_number('１２'), "<@num:12>")

    def test_split_number_2d(self):
        self.assertEqual(NumberHandler.process_number('１３'), '<@num:2d>')
        self.assertEqual(NumberHandler.process_number('１４'), '<@num:2d>')
        self.assertEqual(NumberHandler.process_number('99'), '<@num:2d>')

    def test_split_number_3d(self):
        self.assertEqual(NumberHandler.process_number('100'), '<@num:3d>')
        self.assertEqual(NumberHandler.process_number('500'), '<@num:3d>')
        self.assertEqual(NumberHandler.process_number('999'), '<@num:3d>')

    def test_split_number_4d(self):
        self.assertEqual(NumberHandler.process_number('1000'), '<@num:4d>')
        self.assertEqual(NumberHandler.process_number('5000'), '<@num:4d>')
        self.assertEqual(NumberHandler.process_number('9999'), '<@num:4d>')

    def test_split_number_big(self):
        self.assertEqual(NumberHandler.process_number('１００，０００'), '<@num:big>')
        self.assertEqual(NumberHandler.process_number('50,000'), '<@num:big>')
        self.assertEqual(NumberHandler.process_number('99,999'), '<@num:big>')

    def test_split_number_containing_non_numbers(self):
        self.assertEqual(NumberHandler.process_number('１／４'), '１ _／ _４')
        self.assertEqual(NumberHandler.process_number('CO2'), 'CO _2')
        self.assertEqual(NumberHandler.process_number('Word2vec'), 'Word _2 _vec')
        self.assertEqual(NumberHandler.process_number('ISO14001'), 'ISO _1 _4 _0 _0 _1')       

