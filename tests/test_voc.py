import unittest

import joblib
import numpy as np

from pynkTrombone.voc import Voc, CHUNK


class TestVocComputes(unittest.TestCase):
    def setUp(self):
        self.data = joblib.load('test_data.jbl')
        self.voc = Voc(48000.0)

    def test_basic_voc_output(self):
        out = self.voc.compute(randomize=False)
        np.testing.assert_allclose(out, self.data['basic_voc_output'])

        out2 = self.voc.compute(randomize=False)
        np.testing.assert_allclose(out2, self.data['basic_voc_output_2'], rtol=1e-6, atol=1e-1)

    def test_basic_glottis_output(self):
        self.voc.glottis.update(self.voc.tract.block_time)
        glot = self.voc.glottis.compute(randomize=False)
        self.assertEqual(glot[0], self.data['basic_glot_output'])

    def test_basic_tract_output(self):
        self.voc.glottis.update(self.voc.tract.block_time)
        self.voc.tract.reshape()
        self.voc.tract.calculate_reflections()
        buf = []
        lambda1 = 0
        lambda2 = 0.5 / float(CHUNK)
        glot = self.voc.glottis.compute(randomize=False)

        self.voc.tract.compute(glot[0], lambda1)
        vocal_output_1 = self.voc.tract.lip_output + self.voc.tract.nose_output

        self.voc.tract.compute(glot[0], lambda2)
        vocal_output_2 = vocal_output_1 + self.voc.tract.lip_output + self.voc.tract.nose_output
        buf.append(vocal_output_2 * 0.125)

        self.assertEqual(self.data['basic_tract_output_1'], vocal_output_1)
        self.assertEqual(self.data['basic_tract_output_2'], vocal_output_2)
        self.assertEqual(self.data['basic_buffer_output'], buf[-1])

# class TestVocNumpyRefactors(unittest.TestCase):
#     def setUp(self):
#         self.data = joblib.load('test_data.jbl')
#         self.voc = Voc(48000.0)
#         self.vocnp = Voc(48000.0)
#
#     def test_basic_voc_output(self):
#         out = self.voc.compute(randomize=False)
#         outnp = self.vocnp.compute(randomize=False, use_np=True)
#         np.testing.assert_allclose(out, outnp, rtol=1e-6, atol=1e-6)
#
#         out2 = self.voc.compute(randomize=False)
#         outnp2 = self.vocnp.compute(randomize=False, use_np=True)
#         np.testing.assert_allclose(out2, outnp2, rtol=1e-6, atol=1e-1)
#
#     def test_basic_glottis_output(self):
#         self.voc.glot.update(self.voc.tr.block_time)
#         glot = self.voc.glotnp.compute(randomize=False)
#
#         self.voc.glotnp.update(self.voc.tr.block_time)
#         glotnp = self.voc.glotnp.compute(randomize=False)
#
#         self.assertEqual(glot, glotnp)
#
    # def test_basic_tract_output(self):
    #     self.voc.glot.update(self.voc.tr.block_time)
    #     self.voc.tr.reshape()
    #     self.voc.tr.calculate_reflections()
    #     buf = []
    #     lambda1 = 0
    #     lambda2 = 0.5 / float(CHUNK)
    #     glot = self.voc.glot.compute(lambda1, randomize=False)
    #
    #     self.voc.tr.compute(glot, lambda1)
    #     vocal_output_1 = self.voc.tr.lip_output + self.voc.tr.nose_output
    #
    #     self.voc.tr.compute(glot, lambda2)
    #     vocal_output_2 = vocal_output_1 + self.voc.tr.lip_output + self.voc.tr.nose_output
    #     buf.append(vocal_output_2 * 0.125)
    #
    #     self.assertEqual(self.data['basic_tract_output_1'], vocal_output_1)
    #     self.assertEqual(self.data['basic_tract_output_2'], vocal_output_2)
    #     self.assertEqual(self.data['basic_buffer_output'], buf[-1])


if __name__ == '__main__':
    unittest.main()