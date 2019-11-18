import unittest

import joblib

from voc import Voc, CHUNK
from tract import Tract
from glottis import Glottis

class TestVocComputes(unittest.TestCase):
    def setUp(self):
        self.data = joblib.load('test_data.jbl')
        self.voc = Voc(48000.0)

    def test_basic_voc_output(self):
        out = self.voc.compute(randomize=False)
        self.assertEqual(out, self.data['basic_voc_output'])

        out2 = self.voc.compute(randomize=False)
        self.assertEqual(out2, self.data['basic_voc_output_2'])

    def test_basic_glottis_output(self):
        self.voc.glot.update(self.voc.tr.block_time)
        self.voc.tr.reshape()
        self.voc.tr.calculate_reflections()
        lambda1 = 0
        glot = self.voc.glot.compute(lambda1, randomize=False)
        self.assertEqual(glot, self.data['basic_glot_output'])

    def test_basic_tract_output(self):
        self.voc.glot.update(self.voc.tr.block_time)
        self.voc.tr.reshape()
        self.voc.tr.calculate_reflections()
        buf = []
        lambda1 = 0
        lambda2 = 0.5 / float(CHUNK)
        glot = self.voc.glot.compute(lambda1, randomize=False)

        self.voc.tr.compute(glot, lambda1)
        vocal_output_1 = self.voc.tr.lip_output + self.voc.tr.nose_output

        self.voc.tr.compute(glot, lambda2)
        vocal_output_2 = vocal_output_1 + self.voc.tr.lip_output + self.voc.tr.nose_output
        buf.append(vocal_output_2 * 0.125)

        self.assertEqual(self.data['basic_tract_output_1'], vocal_output_1)
        self.assertEqual(self.data['basic_tract_output_2'], vocal_output_2)
        self.assertEqual(self.data['basic_buffer_output'], buf[-1])

    def test_1(self):
        self.assertTrue(True)

    def test_2(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()