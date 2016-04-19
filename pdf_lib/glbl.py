# global properties
import numpy as np

# r_min and r_max used when reading PDF
rmin = 0.5
rmax = 25.

# q_min and q_max used when calculating PDF
qmin = 0.5
qmax = 25.

# Nynquist step
rstep = np.pi / qmax

# correction factor, so far I just guess
dwFactor = 0.5
bIsoequiv = 0.5

# rows need to be skipped in standard g(r) files
skipRow = 27

# file name
m_id_list_name= 'mid_list.yaml'


class glbl():
    r_min = rmin 
    r_max = rmax
    q_min = qmin
    q_max = qmax
    rstep = rstep
    skip_row = skipRow
    DW_factor = dwFactor
    Bisoequiv = bIsoequiv
    m_id_list = m_id_list_name

glbl = glbl()
