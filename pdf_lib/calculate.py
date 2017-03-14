import os
import time
import yaml
import json
import datetime
import numpy as np
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
    timestring = datetime.datetime.fromtimestamp(\
                 float(timestamp)).strftime('%Y%m%d-%H%M')
    return timestring


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
        #path_str = full_path.split('/')
        #parent_folder_name = path_str[-2]
        #self.stem = parent_folder_name
        stem, tail = os.path.split(full_path)
        self.stem = stem
        self.output_dir = None # overwrite it later
        self.gr_array = None
        self.fail_list = None
        self.r_grid = None
        self.composition_list = None
        self.calculate_params = {}

    def learninglib_build(self, output_dir=None, pdfcal_cfg=None,
                          rdf=True, Bisoequiv=0.1, rstep=None,
                          DebyeCal=False, nosymmetry=False):
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
        Bisoequiv : float, optional
            value of isotropic thermal parameter. default is 0.1.
            scientific equation: Biso = 8 (pi**2) Uiso
        rstep : float, optioanl
            space of PDF. default is pi/100.
        DebyeCal : bool, optional
            option to use Debye calculator. default is False.
        nosymmetry : bool, optional
            DEPRECATED for now. option to apply no symmetry.
            default is False.

        """
        # setup output dir
        timestr = _timestampstr(time.time())
        if output_dir is None:
            tail = "LearningLib_{}".format(timestr)
            output_dir = os.path.join(os.getcwd(), tail)
        print('=== output dir would be {} ==='.format(output_dir))
        self.output_dir = output_dir

        ####### configure pymatgen XRD calculator #####
        # instantiate calculators
        xrd_cal = XRDCalculator()
        self.calculate_params.update({'xrd_wavelength':xrd_cal.wavelength})

        xrd_list = []
        sg_list = []
        composition_list_1 = []
        # (a,b,c, alpha, beta, gamma, volume)
        structure_list_1 = []  # primative cell
        fail_list = []

        ####### configure diffpy PDF calculator ######
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

        # configure calculator
        for k,v in pdfcal_cfg.items():
            setattr(cal, k, v)

        print("==== Parameter used in this PDF calculator is: {} ===="
              .format(pdfcal_cfg))
        print("==== Bisoequiv used in this PDF calculator is: {} ===="
              .format(Bisoequiv))

        # empty list to store results
        gr_list = []
        composition_list_2 = []
        # (a,b,c, alpha, beta, gamma, volume)
        structure_list_2 = []

        ############# loop through cifs #################
        cif_f_list = [ f for f in os.listdir(self.input_dir) if
                       f.endswith('.cif')]
        for cif in cif_f_list:
            _cif = os.path.join(self.input_dir, cif)
            try:
                # diffpy structure
                struc2 = loadStructure(_cif)
                struc2.Bisoequiv = Bisoequiv
                # pymatge structure
                struc = CifParser(_cif).get_structures().pop()

                # calculate PDF/RDF with diffpy
                #if nosymmetry:
                #    (r,g) = cal(nosymmetry(struc), **pdfcal_cfg)
                cal.setStructure(struc2)
                cal.eval()

                # calculate XRD with pymatgen
                xrd = xrd_cal.get_xrd_data(struc)

                # update features
                if rdf:
                    gr_list.append(cal.rdf)
                else:
                    gr_list.append(cal.pdf)
                composition_list_2.append(struc2.composition)
                meta_tuple2 = struc2.lattice.abcABG()+\
                              (struc2.lattice.volume,)
                structure_list_2.append(meta_tuple2)
                print('=== Finished evaluating PDF from structure {} ==='
                       .format(cif))

                xrd_list.append(np.asarray(xrd)[:,:2])
                sg_list.append(struc.get_space_group_info())
                meta_tuple = struc.lattice.abc + struc.lattice.angles\
                             + (struc.volume,)

                structure_list_1.append(meta_tuple)
                composition_list_1.append(struc.composition.as_dict())
                print('=== Finished evaluating XRD from structure {} ==='
                      .format(cif))
            except:
                fail_list.append(cif)

        # finally, store crucial calculation results as attributes
        self.r_grid = cal.rgrid
        self.gr_array = np.asarray(gr_list)/4/np.pi/self.r_grid**2
        self.xrd_info = xrd_list
        self.sg_list = sg_list
        # 1 -> diffpy , 2 -> pymatgen
        self.composition_list_1 = composition_list_1
        self.composition_list_2 = composition_list_2
        self.structure_list_1 = structure_list_1
        self.structure_list_2 = structure_list_2
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
        timestr = _timestampstr(time.time())
        #gr_array_name = '{}_Gr'.format(timestr)
        gr_array_name = 'Gr'
        gr_array_w_name = os.path.join(output_dir, gr_array_name)
        np.save(gr_array_w_name, self.gr_array)

        # rgrid
        #r_grid_name = '{}_rgrid'.format(timestr)
        r_grid_name ='rgrid'
        r_grid_w_name = os.path.join(output_dir, r_grid_name)
        np.save(r_grid_w_name, self.r_grid)

        # sg_list
        #sg_list_name = '{}_sg_list.yml'.format(timestr)
        sg_list_name = 'sg_list.yml'
        sg_list_w_name = os.path.join(output_dir, sg_list_name)
        with open(sg_list_w_name, 'w') as f:
            yaml.dump(self.sg_list, f)

        # xrd_info
        xrd_list_name = 'xrd_info'
        xrd_w_name = os.path.join(output_dir, xrd_list_name)
        np.save(xrd_w_name, self.xrd_info)

        #TODO: simplify saving code
        # composition
        for ind, compo in enumerate([self.composition_list_1,
                                     self.composition_list_2]):
            f_name = "type{}_composition_list.yml".format(ind+1)
            w_name = os.path.join(output_dir,f_name)
            if compo:
                print('INFO: saving {}'.format(w_name))
                with open(w_name, 'w') as f:
                    yaml.dump(compo, f)
            else:
                raise RuntimeError("{} is empty".format(f_name))

        # structure_meta
        for ind, meta in enumerate([self.structure_list_1,
                                    self.structure_list_2]):
            f_name = "type{}_struc_meta.yml".format(ind+1)
            w_name = os.path.join(output_dir, f_name)
            if meta:
                print('INFO: saving {}'.format(w_name))
                with open(w_name, 'w') as f:
                    yaml.dump(meta, f)
            else:
                raise RuntimeError("{} is empty".format(f_name))

        # fail_list
        for ind, meta in enumerate(self.fail_list):
            #f_name = "type{}_fail_list.yml".format(ind+1)
            f_name = "fail_list.yml"
            w_name = os.path.join(output_dir,f_name)
            print('INFO: saving {}'.format(w_name))
            with open(w_name, 'w') as f:
                yaml.dump(meta, f)

        print("======== SUMMARY ======== ")
        print("Number of G(r) calculated is {}"
              .format(np.shape(self.gr_array)[0]))
