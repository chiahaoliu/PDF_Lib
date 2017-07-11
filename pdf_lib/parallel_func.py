import os
import time
import yaml
import json
import datetime
import numpy as np
import pandas as pd
from time import strftime
from pprint import pprint
from uuid import uuid4
import matplotlib.pyplot as plt

from diffpy.Structure import loadStructure
from diffpy.Structure import StructureFormatError
from diffpy.srreal.structureadapter import nosymmetry
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.bondcalculator import BondCalculator

from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from .glbl import pdfCal_cfg, Uiso, bond_range, eps
assert Uiso == 0.005
assert bond_range == 10
assert eps == 0.001

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


def strip_fn(fp):
    """helper function to strip extention of filename"""
    base = os.path.basename(fp)
    fn, ext = os.path.splitext(base)
    return fn

####### configure pymatgen XRD calculator #####
calculate_params = {}

wavelength = 0.5
tth_range = np.arange(0, 90, 0.1)
std_q = theta2q(tth_range, wavelength)

# instantiate calculators
xrd_cal = XRDCalculator()
xrd_cal.wavelength = wavelength
xrd_cal.TWO_THETA_TOL = 10**-2
calculate_params.update({'xrd_wavelength':
                         xrd_cal.wavelength})

#######  PDF calculator instance ######
cal = PDFCalculator()

#######  Bond distance calculator instance ######
bc = BondCalculator(rmax=bond_range)

def map_learninglib(cif_fp, mode='bond_dst'):
    """function designed to build atomic distance list with
    parallel computation

    Parameters
    ----------
    cif_list : str
        full filepath to cif files.
    mode : str, optional
        String to specify quantities being calculated.
        Allowed strings are: ``pdf`` or ``bond_dst``.
        Default to ``pdf``
    """

    ## container for md ##
    rv_dict = {}
    struc_df = pd.DataFrame()
    composition_list_1 = []
    composition_list_2 = []

    ## fields for features ##
    flag = ['primitive', 'ordinary']
    option = [True, False]
    compo_list = [composition_list_1, composition_list_2]
    struc_fields = ['a','b','c','alpha','beta','gamma',
                    'volume', 'sg_label', 'sg_order']

    if mode not in ('pdf', 'bond_dst'):
        raise RuntimeError("Mode must be either 'pdf' or 'bond_dst'")
    try:
        # diffpy structure
        if mode == 'pdf':
            ## calculate PDF/RDF with diffpy ##
            struc = loadStructure(cif_fp)
            struc.Uisoequiv = Uiso
            r_grid, gr = cal(struc, **pdfCal_cfg)
            density = cal.slope
        elif mode == 'bond_dst':
            struc = loadStructure(cif_fp, eps=eps)
            dst = bc(struc)
            uniq_ind = (np.diff(np.r_[-1.0, bc.distances]) > 1e-8)
            uniq_dst = dst[uniq_ind]
            uniq_direction = bc.directions[uniq_ind]
        # pymatgen structure
        struc_meta = CifParser(cif_fp)
        ## test if space group info can be parsed##
        dummy_struc = struc_meta.get_structures(False).pop()
        _sg = dummy_struc.get_space_group_info()
    except:
        # parallelized so direct return
        return os.path.basename(cif_fp)
    else:
        # insert uid
        rv_dict.update({"COD_uid": strip_fn(cif_fp)})
        for f, op, compo in zip(flag, option, compo_list):
            struc = struc_meta.get_structures(op).pop()
            a, b, c = struc.lattice.abc
            aa, bb, cc = struc.lattice.angles
            volume = struc.volume
            sg, sg_order = struc.get_space_group_info()
            for k, v in zip(struc_fields,
                            [a, b, c, aa, bb, cc, volume,
                             sg, sg_order]):
                rv_dict.update({"{}_{}".format(f, k) : v})
            compo_info =dict(struc.composition.as_dict())
            compo.append(compo_info)
            rv_dict.update({"{}_composition".format(f): compo})
        struc_df = struc_df.append(rv_dict, ignore_index=True)

        if mode=='pdf':
            rv_name_list = ['gr', 'density', 'r_grid', 'struc_df']
            print('{:=^80}'.format(' Return '))
            print('\n'.join(rv_name_list))
            return (gr, density, r_grid, struc_df)

        elif mode=='bond_dst':
            rv_name_list = ['uniq_dst', 'uniq_direction', 'struc_df']
            print('{:=^80}'.format(' Return '))
            print('\n'.join(rv_name_list))
            return (uniq_dst, uniq_direction, struc_df)


def learninglib_build(cif_list, xrd=False):
    """function designed for parallel computation

    Parameters
    ----------
    cif_list : list
        List of cif filenames
    xrd : bool, optional
        Wether to calculate xrd pattern. Default to False
    """
    gr_list = []
    density_list = []
    xrd_list = []
    struc_df = pd.DataFrame()
    fail_list = []
    composition_list_1 = []
    composition_list_2 = []

    # database fields
    flag = ['primitive', 'ordinary']
    option = [True, False]
    compo_list = [composition_list_1, composition_list_2]
    struc_fields = ['a','b','c','alpha','beta','gamma',
                    'volume', 'sg_label', 'sg_order']
    # looping
    for _cif in sorted(cif_list):
        try:
            # diffpy structure
            struc = loadStructure(_cif)
            struc.Uisoequiv = Uiso

            ## calculate PDF/RDF with diffpy ##
            r_grid, gr = cal(struc, **pdfCal_cfg)
            density = cal.slope

            # pymatgen structure
            struc_meta = CifParser(_cif)
            ## calculate XRD with pymatgen ##
            if xrd:
                xrd = xrd_cal.get_xrd_data(struc_meta\
                        .get_structures(False).pop())
                _xrd = np.asarray(xrd)[:,:2]
                q, iq = _xrd.T
                q = theta2q(q, wavelength)
                interp_q = assign_nearest(std_q, q, iq)
                xrd_list.append(interp_q)
            else:
                pass
            ## test if space group info can be parsed##
            dummy_struc = struc_meta.get_structures(False).pop()
            _sg = dummy_struc.get_space_group_info()
        #except RuntimeError:  # allow exception to debug
        except:
            print("{} fail".format(_cif))
            fail_list.append(_cif)
        else:
            # no error for both pymatgen and diffpy
            print('=== Finished evaluating PDF from structure {} ==='
                   .format(_cif))
            ## update features ##
            rv_dict = {}
            for f, op, compo in zip(flag, option, compo_list):
                struc = struc_meta.get_structures(op).pop()
                a, b, c = struc.lattice.abc
                aa, bb, cc = struc.lattice.angles
                volume = struc.volume
                sg, sg_order = struc.get_space_group_info()
                for k, v in zip(struc_fields,
                                [a, b, c, aa, bb, cc, volume,
                                 sg, sg_order]):
                    rv_dict.update({"{}_{}".format(f, k) : v})
                compo.append(struc.composition.as_dict())
            struc_df = struc_df.append(rv_dict, ignore_index=True)

            # storing results
            gr_list.append(gr)
            density_list.append(density)

    # end of loop, storing turn result into ndarray
    r_grid = cal.rgrid
    gr_array = np.asarray(gr_list)
    density_array = np.asarray(density_list)
    xrd_info = np.asarray(xrd_list)
    q_grid = std_q

    # talktive statement
    rv_name_list = ['gr_array', 'density_array', 'r_grid',
                    'xrd_info', 'q_grid',
                    'primitive_composition_list',
                    'ordinary_composition_list',
                    'struc_df', 'fail_list']
    print('{:=^80}'.format(' Return '))
    print('\n'.join(rv_name_list))

    rv = gr_array, density_array, r_grid, xrd_info, q_grid,\
         composition_list_1, composition_list_2, struc_df, fail_list

    return rv


# module dict to specify how to unpack rv
RV_LOC_DICT = {'gr': 0, 'density': 1, 'r_grid': 2,
               'xrd': 3, 'q_grid': 4,
               'primitive_compo':5, 'ordinary_compo': 6,
               'struc_df': 7, 'fail_list_map': 0,
               'fail_list_apply': 8}
RV_FN_DICT = {'gr': 'Gr.npy', 'density': 'density.npy',
              'r_grid': 'rgrid.npy', 'xrd': 'xrd.npy',
              'q_grid': 'qgrid.npy', 'struc_df': 'struc_df.csv',
              'fail_list': 'fail_list.json'}

def save_apply_result(apply_rv, output_dir=None):
    """customized function to save result obtained from 'apply' function"""
    timestr = _timestampstr(time.time())
    if output_dir is None:
        tail = "LearningLib_{}".format(timestr)
        output_dir = os.path.join(os.getcwd(), tail)
    os.makedirs(output_dir)
    print('=== output dir would be {} ==='.format(output_dir))

    # organizing results
    _rv = apply_rv[0]

    struc_df = _rv[RV_LOC_DICT['struc_df']]
    primi_compo = _rv[RV_LOC_DICT['primitive_compo']]
    ordin_compo = _rv[RV_LOC_DICT['ordinary_compo']]
    struc_df['primitive_composition'] = primi_compo
    struc_df['ordinary_composition'] = ordin_compo

    w_name = os.path.join(output_dir, RV_FN_DICT['r_grid'])
    np.save(w_name, _rv[RV_LOC_DICT['r_grid']])
    print("INFO: saved {}".format(w_name))

    w_name = os.path.join(output_dir, RV_FN_DICT['gr'])
    np.save(w_name, _rv[RV_LOC_DICT['gr']])
    print("INFO: saved {}".format(w_name))

    w_name = os.path.join(output_dir, RV_FN_DICT['density'])
    np.save(w_name, _rv[RV_LOC_DICT['density']])
    print("INFO: saved {}".format(w_name))

    w_name = os.path.join(output_dir, RV_FN_DICT['struc_df'])
    struc_df.to_csv(w_name)
    print("INFO: saved {}".format(w_name))

    w_name = os.path.join(output_dir, RV_FN_DICT['fail_list'])
    with open(w_name, 'w') as f:
        json.dump(_rv[RV_LOC_DICT['fail_list_apply']], f)
    print("INFO: saved {}".format(w_name))


def join_map_result(map_rv, output_dir=None):
    """customized function to join and save results
    from "map" result"""
    # prepare dir for save
    timestr = _timestampstr(time.time())
    if output_dir is None:
        tail = "LearningLib_{}".format(timestr)
        output_dir = os.path.join(os.getcwd(), tail)
    os.makedirs(output_dir)
    # initialize results being stored
    gr_array = []
    density_list = []
    struc_df = pd.DataFrame()
    fail_list = []
    # NOTE: ignore xrd for now
    # looping and combine results
    for el in map_rv:
        if len(el) != 1:
            gr_array.append(el[RV_LOC_DICT['gr']])
            density_list.append(el[RV_LOC_DICT['density']])
            _df = pd.DataFrame(el[RV_LOC_DICT['struc_df']], copy=True)
            # insert two colums about composition info
            _df['ordinary_composition'] =\
            el[RV_LOC_DICT['ordinary_compo']]
            _df['primitive_composition'] =\
            el[RV_LOC_DICT['primitive_compo']]
            struc_df = struc_df.append(_df, ignore_index=True)
            rgrid = el[RV_LOC_DICT['r_grid']]
        else:
            fail_list.append(el[RV_LOC_DICT['fail_list_map']])
    print("INFO: finish grouping map result")
    # save results
    w_name = os.path.join(output_dir, RV_FN_DICT['r_grid'])
    np.save(w_name, rgrid)
    print("INFO: saved {}".format(w_name))

    gr_ar = np.asarray(gr_array)
    w_name = os.path.join(output_dir, RV_FN_DICT['gr'])
    np.save(w_name, gr_ar)
    print("INFO: saved {}".format(w_name))

    density_ar = np.asarray(density_list)
    w_name = os.path.join(output_dir, RV_FN_DICT['density'])
    np.save(w_name, density_ar)
    print("INFO: saved {}".format(w_name))

    w_name = os.path.join(output_dir, RV_FN_DICT['struc_df'])
    struc_df.to_csv(w_name)
    print("INFO: saved {}".format(w_name))

    w_name = os.path.join(output_dir, RV_FN_DICT['fail_list'])
    with open(w_name, 'w') as f:
        json.dump(fail_list, f)
    print("INFO: saved {}".format(w_name))
