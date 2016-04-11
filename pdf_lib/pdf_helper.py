# helper functions 

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from diffpy.Structure import loadStructure
from pyobjcryst.crystal import CreateCrystalFromCIF
from diffpy.srreal.bondcalculator import BondCalculator
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.pdfcalculator import DebyePDFCalculator

#from pdf_lib.glbl import glbl
from glbl import glbl

def read_full_gr(f_name, rmin=glbl.r_min, rmax = glbl.r_max, skip_num=glbl.skip_row):
    ''' simple function to read .gr data generated from PDFgui'''

    read_in = np.loadtxt(str(f_name), skiprows = skip_num)
    raw_data = np.transpose(read_in)
    raw_data_list = raw_in.tolist()
    upper_ind = raw_list[0].index(rmax)
    lower_ind = raw_list[0].index(rmin)
    cut_data_list = np.asarray([raw_in[0][lower_ind:upper_ind], raw_in[1][lower_ind:upper_ind]])
    return cut_data_list

def read_gr(f_name):
    ''' simple function to read .gr data in database'''
    
    read_in = np.loadtxt(str(f_name))
    plt.plot(read_in[0], read_in[1])
    (gr_name, tail) = os.path.splitext(f_name)
    plt.title(gr_name)
    plt.xlabel('r, A')
    plt.ylabel('G(r), A^-2')

def simple_pdf_cal(input_f, Debye = True, DW_factor = glbl.DW_factor, qmin = glbl.q_min, qmax = glbl.q_max, rmax = glbl.r_max):
    ''' simple pdf calculator. Take input .cif/.xyz file to calculate PDF
        (only use PDFCalculator now, can't calculate nano structrue at this stage)
    
    argument:
        input_f - str - strcuture file name
        Dw_factor - float - value of Debye-Waller factor, which accounts for thermal motions. Default=1 means zero temperature
    '''
    ## calculate theoretical pdf from given structure file

    # create structure
    struc = loadStructure(input_f)

    struc.Bisoequiv = DW_factor
    # calculate PDF
    pdfc = PDFCalculator(qmax = qmax, rmax = rmax)
    dbc = DebyePDFCalculator(qmax = qmax, rmax = rmax)

    if Debye:
        (r, g) = dbc(struc, qmin = qmin)
    else:
        (r, g) = pdfc(struc, qmin=qmin)   
    
    return (r, g)

def dbc_iter(struc_f, iter_range):
    '''change different range'''
    
    import numpy as np
    import matplotlib.pyplot as plt
    from diffpy.srreal.pdfcalculator import DebyePDFCalculator
    from diffpy.Structure import loadStructure
    
    struc = loadStructure(struc_f)
    dbc = DebyePDFCalculator()
    dbc.setStructure(struc)
    dbc.qmax = 20
    dbc.qmin = 0.5
    dbc.rmax = 20
    #par = eval(para)
    #print(dbc.par)
    for step in iter_range:
        (r,g) = dbc(delta2 = step)
        plt.plot(r,g)
    #plt.legend('delta2 =%s' % str(step) )

def iter_bond(x_min, x_max, step=5):
    import numpy as np
    import matplotlib.pyplot as plt
    #from matplotlib.pyplot import plot

    from diffpy.Structure import Structure, Atom, Lattice
    from diffpy.srreal.pdfcalculator import DebyePDFCalculator

    dbc = DebyePDFCalculator()
    dbc.qmax = 20
    dbc.qmin = 0.5
    dbc.rmax = 20
    
    iter_range = np.linspace(x_min, x_max, step)
    fig_dim = len(iter_range)
    
    acs = Atom('Cs', [0, 0, 0])
    acl = Atom('Cl', [0.5, 0.5, 0.5])
    plt.figure()
    for ind, val in enumerate(iter_range):
        cscl = Structure(atoms=[acs, acl],
			lattice=Lattice(val, val, val, 90, 90, 90))
        dbc.setStructure(cscl)
        (r,g) = dbc()
        print(val)
        plt.subplot(fig_dim, 1, ind+1)
        plt.plot(r,g)
        plt.title('bond length = %s' % str(val))

    
def single_plot(x,y):
    ''' simple plot, can't stand the god damn syntax anymore 
    '''
    
    import matplotlib.pyplot as plt
    #plt.figure()
    plt.plot(x,y)
    plt.hold(False)


def multi_plot(data_sets):
    ''' multiplot by calling single_plot'''
    if not isinstance(data_sets, tuple):
        working_data = (data_sets)
    else:
        working_data = data_sets
    
    for data in data_sets:
         x_read = data[0]
         y_read = data[1]
         single_plot(x_read, y_read)
