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
    timestring = datetime.datetime.fromtimestamp(\
                 float(timestamp)).strftime('%Y%m%d-%H%M')
    return timestring

def find_nearest(std_array, val):
    """function to find the index of nearest value"""
    idx = (np.abs(std_array-val)).argmin()
    return idx

def theta2q(theta, wavelength):
    """transform from 2theta to Q(A^-1)"""
    _theta = theta.astype(float)
    rad = np.deg2rad(_theta)
    q = 4*np.pi/wavelength*np.sin(rad/2)
    return q

def assign_nearest(std_array, q_array, iq_val):
    """assign value to nearest grid"""
    idx_list = []
    interp_iq = np.zeros_like(std_array)
    for val in q_array:
        idx_list.append(find_nearest(std_array, val))
    interp_iq[idx_list]=iq_val
    return interp_iq


class PDFLibBuilder:
    ''' a class that loads in .cif in given directory and compute
    corresponding learning lib

    Features will be computed are:
    1. (a, b ,c, alpha, beta, gamma)
    2. chemical composition
    3. RDF (unormalized)
    4. XRD peak positions + intensity

    Label will be computed are:
    1.SpaceGroup Label + space group order
    2.volume of unit cell


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
        print('=== Input dir set to {}, change it if needed ==='
              .format(input_dir))
        self.input_dir = input_dir
        #get the stem
        full_path = os.path.abspath(input_dir)
        stem, tail = os.path.split(full_path)
        self.stem = stem
        self.output_dir = None # overwrite it later
        # diffpy
        self.r_grid = None
        self.gr_array = None
        self.rdf_array = None
        self.density_list = []
        # pymatgen
        self.xrd_array = None
        self.std_q = None

        self.fail_list = None

        self.calculate_params = {}

        cif_list = sorted([ f for f in os.listdir(self.input_dir) if
                              f.endswith('.cif')])
        print("INFO: there are {} structures in input_dir"
              .format(len(cif_list)))
        self.cif_list = cif_list

    def learninglib_build(self, output_dir=None, pdfcal_cfg=None,
                          rdf=True, xrd=False, Bisoequiv=0.1,
                          rstep=None, DebyeCal=False, nosymmetry=False,
                          tth_range=None, wavelength=0.5):
        """ method to build learning lib with diffpy based on path
        of cif library. Paramters of G(r) calculation are set
        via glbl.<attribute>. "PDFCal_config.txt" file with PDFCalculator
        configuration will also be output

        Parameters
        ----------
        pdfcal_cfg : dict, optional
            configuration of PDF calculator, default is the one defined
            inside glbl class.
        rdf : bool, optional
            option to compute RDF or not. default to True, if not,
            compute pdf
        xrd : bool, optional
            option to compute XRD (which is slow). default to False.
        Bisoequiv : float, optional
            value of isotropic thermal parameter. default is 0.1.
            scientific equation: Biso = 8 (pi**2) Uiso
        rstep : float, optioanl
            space of PDF. default is pi/qmax.
        DebyeCal : bool, optional
            option to use Debye calculator. default is False.
        nosymmetry : bool, optional
            DEPRECATED for now. option to apply no symmetry.
            default is False.
        tth_range : ndarray, optional
            range of 2theta. default is [0:0.1:90]
        wavelength : float, optional
            wavelength in angstroms, default to 0.5 A which corresponds
            to Qmax ~= 17
        """
        # setup output dir
        timestr = _timestampstr(time.time())
        if output_dir is None:
            tail = "LearningLib_{}".format(timestr)
            output_dir = os.path.join(os.getcwd(), tail)
        print('=== output dir would be {} ==='.format(output_dir))
        self.output_dir = output_dir
        if tth_range is None:
            self.tth_range = np.arange(0, 90, 0.1)
        self.wavelength = wavelength
        self.std_q = theta2q(self.tth_range, self.wavelength)

        ####### configure pymatgen XRD calculator #####
        # instantiate calculators
        xrd_cal = XRDCalculator()
        xrd_cal.wavelength = self.wavelength
        xrd_cal.TWO_THETA_TOL = 10**-2
        self.calculate_params.update({'xrd_wavelength':
                                       xrd_cal.wavelength})

        xrd_list = []
        sg_list = []
        # (a,b,c, alpha, beta, gamma, volume)
        structure_list_1 = []  # primative cell
        structure_list_2 = []  # ordinary cell
        # chemical element
        composition_list_1 = []  # primative cell
        composition_list_2 = []  # ordinary cell
        fail_list = []

        ####### configure diffpy PDF calculator ######
        if DebyeCal:
            cal = DebyePDFCalculator()
            self.calculator_type = 'Debye'
        cal = PDFCalculator()
        self.calculator = cal
        self.calculator_type = 'PDF'
        self.calculate_params.update({'calculator_type':
                                      self.calculator_type})
        # setup calculator parameters
        if rstep is None:
            rstep = glbl.rstep
        self.rstep = rstep
        self.calculator.rstep = rstep  # annoying fact
        self.calculate_params.update({'rstep':rstep})

        if pdfcal_cfg is None:
            self.pdfcal_cfg = glbl.cfg
        self.calculate_params.update(self.pdfcal_cfg)

        # configure calculator
        for k,v in self.pdfcal_cfg.items():
            setattr(self.calculator, k, v)
        # empty list to store results
        gr_list = []
        rdf_list = []
        print("====== INFO: calculation parameters:====\n{}"
              .format(self.calculate_params))
        struc_df = pd.DataFrame()
        ############# loop through cifs #################
        for cif in sorted(self.cif_list):
            _cif = os.path.join(self.input_dir, cif)
            try:
                # diffpy structure
                struc = loadStructure(_cif)
                struc.Bisoequiv = Bisoequiv

                ## calculate PDF/RDF with diffpy ##
                if nosymmetry:
                    struc = nosymmetry(struc)
                cal.setStructure(struc)
                cal.eval()

                # pymatge structure
                struc_meta = CifParser(_cif)
                ## calculate XRD with pymatgen ##
                if xrd:
                    xrd = xrd_cal.get_xrd_data(struc_meta\
                            .get_structures(False).pop())
                    _xrd = np.asarray(xrd)[:,:2]
                    q, iq = _xrd.T
                    interp_q = assign_nearest(self.std_q, q, iq)
                    xrd_list.append(interp_q)
                else:
                    pass
                ## test space group info ##
                _sg = struc_meta.get_structures(False).pop()\
                        .get_space_group_info()
            except:
                print("{} fail".format(_cif))
                fail_list.append(cif)
            else:
                # no error for both pymatgen and diffpy
                gr_list.append(cal.pdf)
                rdf_list.append(cal.rdf)
                self.density_list.append(cal.slope)
                print('=== Finished evaluating PDF from structure {} ==='
                       .format(cif))
                ## update features ##
                flag = ['primitive', 'ordinary']
                option = [True, False]
                compo_list = [composition_list_1, composition_list_2]
                struc_fields = ['a','b','c','alpha','beta','gamma', 'volume']
                for f, op, compo in zip(flag, option, compo_list):
                    rv_dict = {}
                    struc = struc_meta.get_structures(op).pop()
                    a, b, c = struc.lattice.abc
                    aa, bb, cc = struc.lattice.angles
                    volume = struc.volume
                    for k, v in zip(struc_fields,
                                    [a, b, c, aa, bb, cc, volume]):
                        rv_dict.update({"{}_{}".format(f, k) : v})
                    compo.append(struc.composition.as_dict())
                    struc_df = struc_df.append(rv_dict,
                                               ignore_index=True)
                # sg info, use the ordinary setup
                sg_list.append(struc.get_space_group_info())
                print('=== Finished evaluating XRD from structure {} ==='
                      .format(cif))

        # finally, store crucial calculation results as attributes
        self.r_grid = cal.rgrid
        #4*pi * r^2 * rho(r) = R(r)  -> RDF to density 
        self.gr_array = np.asarray(gr_list)/4/np.pi/self.r_grid**2
        self.rdf_array = np.asarray(gr_list)
        self.density_list = np.asarray(self.density_list)
        self.xrd_info = np.asarray(xrd_list)
        self.sg_list = sg_list
        # 1 -> primitive , 2 -> ordinary
        self.composition_list_1 = composition_list_1
        self.composition_list_2 = composition_list_2
        self.struc_df = struc_df
        self.fail_list = fail_list

    def save_data(self):
        """ a method to save outputs """
        output_dir = self.output_dir
        _makedirs(output_dir)
        # save config of calculator
        with open(os.path.join(output_dir, \
                  'learninglib_config.txt'), 'w') as f:
            para_dict = dict(self.calculate_params)
            f.write(str(para_dict))

        # save gr, r, composition and fail list
        gr_array_name = 'Gr'
        gr_array_w_name = os.path.join(output_dir, gr_array_name)
        np.save(gr_array_w_name, self.gr_array)

        # rgrid
        r_grid_name ='rgrid'
        r_grid_w_name = os.path.join(output_dir, r_grid_name)
        np.save(r_grid_w_name, self.r_grid)

        # std_q
        q_grid_name = 'qgrid'
        q_grid_w_name = os.path.join(output_dir, q_grid_name)
        np.save(q_grid_w_name, self.std_q)

        # density_list
        f_name = 'density'
        w_name = os.path.join(output_dir, f_name)
        np.save(w_name, self.density_list)

        # sg_list
        sg_list_name = 'sg_list.json'
        sg_list_w_name = os.path.join(output_dir, sg_list_name)
        with open(sg_list_w_name, 'w') as f:
            json.dump(self.sg_list, f)

        # xrd_info
        xrd_list_name = 'xrd_info'
        xrd_w_name = os.path.join(output_dir, xrd_list_name)
        np.save(xrd_w_name, self.xrd_info)

        #TODO: simplify saving code
        # composition
        fn_stem_list = ['primitive', 'ordinary']
        for ind, compo in enumerate([self.composition_list_1,
                                     self.composition_list_2]):
            f_name = "{}_composition_list.json".format(fn_stem_list[ind])
            w_name = os.path.join(output_dir,f_name)
            if compo:
                print('INFO: saving {}'.format(w_name))
                with open(w_name, 'w') as f:
                    json.dump(compo, f)
            else:
                raise RuntimeError("{} is empty".format(f_name))

        # structure_meta
        f_name = "struc_df.json"
        w_name = os.path.join(output_dir, f_name)
        self.struc_df.to_json(w_name)

        # fail_list
        f_name = "fail_list.json"
        w_name = os.path.join(output_dir,f_name)
        print('INFO: saving {}'.format(w_name))
        with open(w_name, 'w') as f:
            json.dump(meta, f)

        print("======== SUMMARY ======== ")
        print("Number of fature calculated is {}"
              .format(np.shape(self.gr_array)[0]))
