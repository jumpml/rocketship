#  JumpML Rocketship - Neural Network Inference with Audio Processing
#  
#  Copyright 2020-2024 JUMPML LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  gen_tanh_table.py
# 
import numpy as np
import matplotlib.pyplot as plt
from convert_model import printHeader, printVector, convert_datatype

# TABLE_SIZE = 201 # Choose powers of 2 so that indexing into table is easy
MAX_XRANGE = 8.0   # 16-bit fixed-point saturates beyond 6
DELTA_X = 1/8      # DESIGN PARAMETER
SCALE_FAC = int(1.0/DELTA_X)
TABLE_X = np.arange(0, MAX_XRANGE+1e-7, DELTA_X)
TABLE_SIZE = len(TABLE_X)
TANH_TABLE = np.tanh(TABLE_X, dtype=np.float32)

def dump_table_Cfile(x, y, fname="include/tanh_table.h", datatype='int16_t', postfix=""):
    f = open(fname, 'w')
    table_len = len(y)
    printHeader(f, include_files=["<stdint.h>", "common_def.h"],fName=fname, autogen_str="by gen_tanh_table.py")

    if datatype == 'float':
        typeStr = 'f'
    else:
        typeStr = ''
    
    f.write(f'\n#define TANH_TABLE_SIZE{postfix} {table_len}\n')
    f.write(f'#define TANH_TABLE_MAXINDEX{postfix} {table_len-1}\n')
    f.write(f'#define TANH_DELTAX{postfix} {convert_datatype(DELTA_X, datatype=datatype, nFracBits=15)}{typeStr}\n')
    f.write(f'#define TANH_SCALEFAC{postfix} {SCALE_FAC}\n')
    f.write('\n\n')
    printVector(f, y, f'tanh_table{postfix}', datatype=datatype, nFracBits=15) # ignores nFracBits when datatype is float
   
def tanh_approx(x):
    sign_x = np.sign(x)
    abs_x = sign_x * x 
    idx = np.floor(0.5 + abs_x * SCALE_FAC)
    idx = idx.astype(int)
    idx = np.maximum(0, np.minimum(TABLE_SIZE-1, idx))
    dx = abs_x - DELTA_X * idx
    y = TANH_TABLE[idx]
    dy = 1-y*y
    y = y + dx*dy*(1-y*dx)
    return sign_x*y




def validate_function():
    x = np.linspace(-MAX_XRANGE, MAX_XRANGE, 1000)
    y_ref = np.tanh(x, dtype=np.float32)
    y = tanh_approx(x)
    abs_err = np.abs(y-y_ref)
    plt.plot(x,abs_err)
    print(f'MAE={np.mean(abs_err)} \t Max Err = {np.max(abs_err)}')
    # plt.plot(x,y)
    # plt.plot(x,y_ref)
    plt.show()




if __name__ == '__main__':
    dump_table_Cfile(TABLE_X, TANH_TABLE, datatype='float')
    dump_table_Cfile(TABLE_X, TANH_TABLE, fname ="include/tanh_table_S16.h", datatype='int16_t',postfix="_S16")
    validate_function()

    # plt.plot(y  - tansig_table)
    # # plt.plot(tansig_table)
    # plt.show()
