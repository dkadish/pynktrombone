import joblib

from voc import Voc, CHUNK

v = Voc(48000.0)
voc_out = v.compute(randomize=False)
voc_out2 = v.compute(randomize=False)

# Restart
v = Voc(48000.0)
v.glottis.update(v.tract.block_time)
v.tract.reshape()
v.tract.calculate_reflections()
buf = []
lambda1 = 0
lambda2 = 0.5/float(CHUNK)
glot = v.glottis.compute(randomize=False)

v.tract.compute(glot, lambda1)
vocal_output_1 = v.tract.lip_output + v.tract.nose_output

v.tract.compute(glot, lambda2)
vocal_output_2 = vocal_output_1 + v.tract.lip_output + v.tract.nose_output
buf.append(vocal_output_2 * 0.125)

# Full Chunk


test_values = {
    'basic_voc_output': voc_out,
    'basic_voc_output_2': voc_out2,
    'basic_glot_output': glot,
    'basic_tract_output_1': vocal_output_1,
    'basic_tract_output_2': vocal_output_2,
    'basic_buffer_output': buf[-1]
               }
joblib.dump(test_values, './test_data.jbl')
