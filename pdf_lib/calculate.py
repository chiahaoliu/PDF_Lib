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

from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# pymatgen syntax
"""
struc = CifParser(<fp>).get_structures().pop()
meta_tuple = struc.lattice.abc + aa.lattice.angles
volume = struc.volume
spacegroup_info = struc.get_space_group_info()
element_info = struc.species  # a list
"""

#from pdf_lib.glbl import glbl
from .glbl import glbl


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
        print('=== Input dir set to {}, change it if needed ==='.format(input_dir))
        self.input_dir = input_dir
        #get the stem
        full_path = os.path.abspath(input_dir)
        path_str = full_path.split('/')
        parent_folder_name = path_str[-2]
        self.stem = parent_folder_name
        self.output_dir = None # overwrite it later
        self.gr_array = None
        self.fail_list = None
        self.r_grid = None
        self.composition_list = None
        self.calculate_params = {}

    def gr_lib_build(self, output_dir=None, pdfcal_cfg=None,
                     rdf=True, Bisoequiv=0.1, rstep=None,
                     DebyeCal=False, nosymmetry=False):
        """ method to calculate G(r) based on path of cif library located at.

        Paramters of G(r) calculation are set via glbl.<attribute>.
        After entire method, .npy file contains all G(r), space_group_symbol
        and material name will be saved respectively

        Parameters
        ----------
        output_dir : str, optional
            optional. path to lib of cif files. 
            Default is "{inputdir}/PDFLib_{time}/"
            "PDFCal_config.txt" file with PDFCalculator
            configuration will also be saved in output directory
        pdfcal_cfg : dict, optional
            configuration of PDF calculator, default is the one defined
            inside glbl class.
        rdf : bool, optional
            option to return RDF or not. default to True, if not,
            return pdf
        Bisoequiv : float, optional
            value of isotropic thermal parameter. default is 0.1.
            scientific equation: Biso = 8 (pi**2) Uiso
        rstep : float, optioanl
            space of PDF. default is pi/100.
        DebyeCal : bool, optional
            option to use Debye calculator. default is False.
        nosymmetry : bool, optional
            option to apply no symmetry. default is False.

        """
        timestr = _timestampstr(time.time())
        if output_dir is None:
            tail = "PDF_{}".format(timestr)
            output_dir = os.path.join(os.getcwd(), tail)
        print('=== output dir would be {} ==='.format(output_dir))
        self.output_dir = output_dir

        # instantiate calculator
        if DebyeCal:
            cal = DebyePDFCalculator()
            self.calculator_type = 'Debye'
        cal = PDFCalculator()
        self.calculator_type = 'PDF'
        self.calculate_params.update({'calculator_type':
                                      self.calculator_type})

        # setup calculator parameters
        if rstep is None:
            rstep = glbl.rstep
        self.rstep = rstep
        self.calculate_params.update({'rstep':rstep})

        if pdfcal_cfg is None:
            pdfcal_cfg = glbl.cfg
        self.calculate_params.update(pdfcal_cfg)

        print("==== Parameter used in this PDF calculator is: {} ===="
              .format(pdfcal_cfg))
        print("==== Bisoequiv used in this PDF calculator is: {} ===="
              .format(Bisoequiv))
        # list cif dir
        cif_f_list = [ f for f in os.listdir(self.input_dir) if
                       os.path.isfile(os.path.join(self.input_dir, f))]
        gr_list = []
        composition_list = []
        fail_list = []
        # configure calculator
        for k,v in pdfcal_cfg.items():
            setattr(cal, k, v)
        for cif in cif_f_list:
            # calculate PDF with diffpy
            try:
                struc = loadStructure(os.path.join(self.input_dir, cif))
                struc.Bisoequiv =  Bisoequiv
                #if nosymmetry:
                #    (r,g) = cal(nosymmetry(struc), **pdfcal_cfg)
                cal.setStructure(struc)
                cal.eval()

                print('=== Finished evaluating structure {} ==='
                       .format(cif))
                if rdf:
                    gr_list.append(cal.rdf)
                else:
                    gr_list.append(cal.pdf)
                gr_list.append(g)
                composition_list.append(struc.composition)
                self.r_grid = r
            except: # too many unexpected errors from open data base
                fail_list.append(cif)
                pass
        # set attributes
        self.gr_array = np.asarray(gr_list)
        self.fail_list = fail_list
        self.r_grid = r
        self.composition_list = composition_list

    def save_data(self):
        """ a method to save outputs """
        output_dir = self.output_dir
        _makedirs(output_dir)
        # save config of calculator
        with open(os.path.join(output_dir, 'PDFCal_config.txt'), 'w') as f:
            para_dict = dict(self.calculate_params)
            f.write(str(para_dict))
        # save gr, r, composition and fail list
        timestr = _timestampstr(time.time())
        gr_array_name = '{}_{}_Gr'.format(self.stem, timestr)
        gr_array_w_name = os.path.join(output_dir, gr_array_name)

        np.save(gr_array_w_name, self.gr_array)

        r_grid_name = '{}_{}_rgrid'.format(self.stem, timestr)
        r_grid_w_name = os.path.join(output_dir, r_grid_name)
        np.save(r_grid_w_name, self.r_grid)

        composition_list_name = "{}_{}_composition_list.yaml"\
                                .format(self.stem, timestr)
        composition_list_w_name= os.path.join(output_dir,
                                              composition_list_name)
        #np.savetxt(composition_list_w_name, self.composition_list, fmt="%s")
        with open(composition_list_w_name, 'w') as f:
            yaml.dump(self.composition_list, f)

        fail_list_name = '{}_{}_fail_list.yaml'.format(self.stem, timestr)
        fail_list_w_name = os.path.join(output_dir,
                                        fail_list_name)
        #np.savetxt(fail_list_w_name, self.fail_list, fmt="%s")
        with open(fail_list_w_name, 'w') as f:
            yaml.dump(self.fail_list, f)

        print("======== SUMMARY ======== ")
        print("Number of G(r) calculated is {}"
              .format(np.shape(self.gr_array)[0]))
