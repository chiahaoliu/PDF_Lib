# global properties
import numpy as np


# r_min and r_max used when reading PDF
rmin = 0.5
rmax = 50. # extend to longer region, chop later

# q_min and q_max used when calculating PDF
qmin = 0.5
qmax = 25.

# Nynquist step
rstep = (np.pi / qmax) - 1E-4

# Qdamp value; use that one from XPD
qdamp = 0.04
qbroad = 0.04
delta2 = 2.3

pdfCal_cfg = dict(rmin=rmin, rmax=rmax, qmin=qmin, qmax=qmax,
                  qdamp=qdamp, qbroad=qbroad, rstep=rstep,
                  delta2=delta2)

# correction factor; used that ones from XPD
Uiso = 0.005

# rows need to be skipped in standard g(r) files
skipRow = 27

# rmax for bond distance calculation
bond_range = 15

# tolorance of atom positions
eps = 0.001
