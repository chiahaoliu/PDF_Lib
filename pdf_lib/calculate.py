
import os
import time
import yaml
import datetime
import linecache
import numpy as np
import pandas as pd
from time import strftime
import matplotlib.pyplot as plt

from diffpy.Structure import loadStructure
from diffpy.Structure import StructureFormatError
from diffpy.srreal.structureadapter import nosymmetry
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from diffpy.srreal.pdfcalculator import PDFCalculator

#from pdf_lib.glbl import glbl
from glbl import glbl


def _makedirs(path_name):
    '''function to support python2 stupid logic'''
    if os.path.isdir(path_name):
        pass
    else:
        os.makedirs(path_name)


def _timestampstr(timestamp):
    ''' convert timestamp to strftime formate '''
    timestring = datetime.datetime.fromtimestamp(float(timestamp)).strftime('%Y%m%d-%H%M')
    return timestring


class PDF_cal:
    ''' a class that loads in .cif in given directory, compute PDF and
    output all PDFs as a numpy array

    Parameters:
    -----------
    input_dir : str
        optional. path to where you stores cif files. default is current
        directory.
    '''
    def __init__(self, input_dir=None):
        # set up API_key
        if input_dir is None:
            input_dir = os.getcwd()
        self.input_dir = input_dir
        self.output_dir = None # overwrite it later

    def gr_lib_build(self, output_dir=None, DebyeCal=False,
                     nosymmetry=False):
        ''' method to calculate G(r) based on path of cif library located at.

        Paramters of G(r) calculation are set via glbl.<attribute>.
        After entire method, .npy file contains all G(r), space_group_symbol
        and material name will be saved respectively

        Parameters
        ----------
        output_dir : str
            optional. path to lib of cif files. Default is
            "self.inputdir/PDFLib_{time}_{parameter}/".
        DebyeCal : bool
            option to use Debye calculator. default is False.
        nosymmetry : bool
            option to apply no symmetry. default is False.

        '''
        timestr = _timestampstr(time.time())
        if output_dir is None:
            tail = "PDF_{}_{}".format(timestr, glbl.cfg)
            output_dir = os.path.join(self.input_dir, tail)
        self.output_dir = output_dir
        _makedirs(self.output_dir)

        # set up calculation environment
        if DebyeCal:
            cal = DebyePDFCalculator()
        cal = PDFCalculator()
        cal.rstep = glbl.rstep
        cfg = glbl.cfg
        Bisoequiv = glbl.Bisoequiv
        print("====Parameter used in this PDF calculator is: {}===="
              .format(cfg))
        print("====Bisoequiv used in this PDF calculator is: {}===="
              .format(Bisoequiv))

        # step 1: list cif dir

        cif_f_list = [ f for f in os.listdir(self.input_dir)]
        gr_list = []
        composition_list = []
        fail_list = []
        for cif in cif_f_list:
            # part 2: calculate PDF with diffpy
            try:
                struc = loadStructure(os.path.join(self.input_dir, cif))
                struc.Bisoequiv =  Bisoequiv
                if nosymmetry:
                    (r,g) = cal(nosymmetry(struc), **cfg)
                (r,g) = cal(struc, **cfg)
                print('Finished calculation of G(r) on {}'.format(cif))
                gr_list.append(g)
                composition_list.append(struc.composition)
            except: # too many unexpected errors from open data base
                fail_list.append(cif)
                pass
        gr_len = len(g)
        gr_list_len = len(gr_list)
        gr_array = np.asarray(gr_list)
        gr_array.resize(gr_list_len/gr_len, gr_len)
        gr_array_name = '{}_Gr'.format(timestr)
        gr_array_w_name = os.path.join(self.output_dir, gr_list_name)
        print('Saving {}'.format(gr_array_w_name))
        np.save(gr_array_w_name, gr_array)
        del gr_list

        r_grid_name = '{}_rgrid'.format(time_str)
        r_grid_w_name = os.path.join(output_dir, r_grid_name)
        np.save(r_grid_w_name, r)

        composition_list_name = '{}_composition'.format(timestr)
        composition_list_w_name= os.path.join(output_dir,
                                              composition_list_w_name)
        np.savetxt(composition_list_w_name, composition_list, fmt="%s")

        print("======== SUMMARY ======== ")
        print("Number of G(r) calculated is {}"
              .format(np.shape(gr_array)[0]))
        return gr_array
