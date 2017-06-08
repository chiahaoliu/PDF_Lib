import os
import time
import yaml
import json
import datetime
import numpy as np
import pandas as pd
from time import strftime
from pprint import pprint
import matplotlib.pyplot as plt

from diffpy.Structure import loadStructure
from diffpy.Structure import StructureFormatError
from diffpy.srreal.structureadapter import nosymmetry
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from diffpy.srreal.pdfcalculator import PDFCalculator

from pdf_lib.glbl import pdfCal_cfg, Uiso, delta2

cal = PDFCalculator()
cal.delta2 = delta2
ni = loadStructure('ni.cif')
ni.Uisoequiv = Uiso
nacl = loadStructure('1000041.cif')
nacl.Uisoequiv = Uiso
print("Uiso = {}".format(Uiso))


def qdamp_test(struc, rmax_val, qdamp_array=None):
    pdfCal_cfg['rmax'] = rmax_val
    N = qdamp_array.size
    fig, ax = plt.subplots(N, figsize=(20, 6),
                           sharex=True, sharey=True)
    for _ax, qdamp_val in zip(ax, qdamp_array):
        pdfCal_cfg['qdamp'] = qdamp_val
        r, g = cal(struc, **pdfCal_cfg)
        #cal.setStructure(struc)
        #for k, v in pdfCal_cfg.items():
        #    setattr(cal, k, v)
        #cal.eval()
        #r = cal.rgrid
        #g = cal.pdf
        _ax.plot(r, g, linestyle='-',
                label="{:.3f}".format(qdamp_val))
        _ax.legend()
    fig.suptitle('{}'.format(struc.composition))
    print(cal.rstep)
    print(cal.slope)

qdamp_array = np.arange(0, 0.1, 0.02)
rmax = 50.
qdamp_test(ni, rmax, qdamp_array)
qdamp_test(nacl, rmax, qdamp_array)

