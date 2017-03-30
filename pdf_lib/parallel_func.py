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

calculate_params = {}

wavelength = 0.5
tth_range = np.arange(0, 90, 0.1)
std_q = theta2q(tth_range, wavelength)

####### configure pymatgen XRD calculator #####
# instantiate calculators
xrd_cal = XRDCalculator()
xrd_cal.wavelength = wavelength
xrd_cal.TWO_THETA_TOL = 10**-2
calculate_params.update({'xrd_wavelength':
                         xrd_cal.wavelength})

####### configure diffpy PDF calculator ######
cal = PDFCalculator()
# setup calculator parameters
rstep = glbl.rstep
cal.rstep = rstep  # annoying fact
calculate_params.update({'rstep':rstep})
Bisoequiv = 0.1


pdfcal_cfg = glbl.cfg
calculate_params.update(pdfcal_cfg)

# configure calculator
for k,v in pdfcal_cfg.items():
    setattr(cal, k, v)


def map_learninglib(cif_path, xrd=False):
    _cif = cif_path
    sg_list = []
    fail_list = []
    struc_df = pd.DataFrame()
    composition_list_1 = []
    composition_list_2 = []
    try:
        # diffpy structure
        struc = loadStructure(_cif)
        struc.Bisoequiv = Bisoequiv

        ## calculate PDF/RDF with diffpy ##
        cal.setStructure(struc)
        cal.eval()

        # pymatge structure
        struc_meta = CifParser(_cif)
        ## calculate XRD with pymatgen ##
        """
        if xrd:
            xrd = xrd_cal.get_xrd_data(struc_meta\
                    .get_structures(False).pop())
            _xrd = np.asarray(xrd)[:,:2]
            q, iq = _xrd.T
            interp_q = assign_nearest(std_q, q, iq)
            xrd_list.append(interp_q)
        else:
            pass
        """
        ## test space group info ##
        _sg = struc_meta.get_structures(False).pop()\
                .get_space_group_info()
    except:
        print("{} fail".format(_cif))
        fail_list.append(_cif)
    else:
        # no error for both pymatgen and diffpy
        gr = cal.pdf
        rdf = cal.rdf
        density = cal.slope
        print('=== Finished evaluating PDF from structure {} ==='
               .format(_cif))
        ## update features ##
        flag = ['primitive', 'ordinary']
        option = [True, False]
        compo_list = [composition_list_1, composition_list_2]
        struc_fields = ['a','b','c','alpha','beta','gamma',
                        'volume', 'sg_label']
        for f, op, compo in zip(flag, option, compo_list):
            rv_dict = {}
            struc = struc_meta.get_structures(op).pop()
            a, b, c = struc.lattice.abc
            aa, bb, cc = struc.lattice.angles
            volume = struc.volume
            sg = struc.get_space_group_info()
            for k, v in zip(struc_fields,
                            [a, b, c, aa, bb, cc, volume, sg]):
                rv_dict.update({"{}_{}".format(f, k) : v})
            compo.append(struc.composition.as_dict())
            struc_df = struc_df.append(rv_dict,
                                       ignore_index=True)
        # sg info, use the ordinary setup
        sg_list.append(struc.get_space_group_info())
        print('=== Finished evaluating XRD from structure {} ==='
              .format(_cif))

        # finally, store crucial calculation results as attributes
        r_grid = cal.rgrid
        #4*pi * r^2 * rho(r) = R(r)  -> RDF to density 
        gr_array = np.asarray(gr)/4/np.pi/r_grid**2
        rdf_array = np.asarray(rdf)
        density = np.asarray(density)
        #xrd_info = np.asarray(xrd_list)
        # 1 -> primitive , 2 -> ordinary

        print("Return : gr_array, rdf_array, density_list, "
              "sg_list, composition_list_1, composition_list_2, struc_df, "
              "fail_list")

        return gr_array, rdf_array, density, sg_list,\
               composition_list_1, composition_list_2, struc_df, fail_list


def learninglib_build(cif_list, input_dir, xrd=False):
    print("====== INFO: calculation parameters:====\n{}"
          .format(calculate_params))
    gr_list = []
    rdf_list = []
    density_list = []
    struc_df = pd.DataFrame()
    xrd_list = []
    sg_list = []
    composition_list_1 = []  # primative cell
    composition_list_2 = []  # ordinary cell
    fail_list = []
    ############# loop through cifs #################
    for cif in sorted(cif_list):
        _cif = os.path.join(input_dir, cif)
        try:
            # diffpy structure
            struc = loadStructure(_cif)
            struc.Bisoequiv = Bisoequiv

            ## calculate PDF/RDF with diffpy ##
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
                interp_q = assign_nearest(std_q, q, iq)
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
            density_list.append(cal.slope)
            print('=== Finished evaluating PDF from structure {} ==='
                   .format(cif))
            ## update features ##
            flag = ['primitive', 'ordinary']
            option = [True, False]
            compo_list = [composition_list_1, composition_list_2]
            struc_fields = ['a','b','c','alpha','beta','gamma',
                            'volume', 'sg_label']
            for f, op, compo in zip(flag, option, compo_list):
                rv_dict = {}
                struc = struc_meta.get_structures(op).pop()
                a, b, c = struc.lattice.abc
                aa, bb, cc = struc.lattice.angles
                volume = struc.volume
                sg = struc.get_space_group_info()
                for k, v in zip(struc_fields,
                                [a, b, c, aa, bb, cc, volume, sg]):
                    rv_dict.update({"{}_{}".format(f, k) : v})
                compo.append(struc.composition.as_dict())
                struc_df = struc_df.append(rv_dict,
                                           ignore_index=True)
            # sg info, use the ordinary setup
            sg_list.append(struc.get_space_group_info())
            print('=== Finished evaluating XRD from structure {} ==='
                  .format(cif))

    # finally, store crucial calculation results as attributes
    r_grid = cal.rgrid
    #4*pi * r^2 * rho(r) = R(r)  -> RDF to density 
    gr_array = np.asarray(gr_list)/4/np.pi/r_grid**2
    rdf_array = np.asarray(rdf_list)
    density_list = np.asarray(density_list)
    xrd_info = np.asarray(xrd_list)
    # 1 -> primitive , 2 -> ordinary

    print("Return : gr_array, rdf_array, density_list, xrd_info, "
          "sg_list, composition_list_1, composition_list_2, struc_df, "
          "fail_list")

    return gr_array, rdf_array, density_list, xrd_info, sg_list,\
           composition_list_1, composition_list_2, struc_df, fail_list


def save_data(rv, output_dir=None):
    # setup output dir
    timestr = _timestampstr(time.time())
    if output_dir is None:
        tail = "LearningLib_{}".format(timestr)
        output_dir = os.path.join(os.getcwd(), tail)
    print('=== output dir would be {} ==='.format(output_dir))
    f_name_list = ['Gr.npy', 'rdf.npy', 'density.npy', 'xrd_info.npy',
                   'sg_list.json', 'primitive_composition.json',
                   'ordinary_composition.json', 'struc_df.json',
                   'fail_list.json']
    for el, f_name in zip(rv, f_name_list):
        w_name = os.path.join(output_dir, f_name)
        if f_name.endswith('.npy'):
            np.save(w_name, el)
        elif f_name.endswith('.json'):
            with open(w_name, 'w') as f:
                json.dump(el, f)
        print("INFO: saved {}".format(w_name))
