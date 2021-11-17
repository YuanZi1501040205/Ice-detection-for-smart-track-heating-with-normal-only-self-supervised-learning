'''
Parameters of this project.
This file is used for storing the parameters that would
not be changed frequently.
'''

# Lengths of input and output traces
num_input = 2500
num_output = 2500

# Batch size
batch_size = 16

# Zoom range, should be in the range of (1, num_output)
zoom_range = (500, 1000)

# Input and output frequency number
num_input_complex = 150 * 2
num_output_complex = 46 * 2

# The index of 1~40Hz
freq_range_1_40 = (5, 201)

# The index of 1-10Hz errors from 1-40Hz
freq_range_1_10 = (0, 56)

# Time step
time_step = 0.002

# GPU number, if no GPU or only one GPU, should use None instead of a number.
gpu_number = 2
