import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import pandas as pd
import numpy as np
from pgmpy.estimators.CITests import g_sq

# P(A|C)=0.6, P(B|C)=0.4, P(C)=0.5
data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
data['E'] = data['A'] + data['B'] + data['C']
print(g_sq(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05))