import os
import time
import datetime
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen import MPRester
from pymatgen.io.cif import CifWriter
from diffpy.Structure import loadStructure
from diffpy.srreal.structureadapter import nosymmetry
from diffpy.srreal.pdfcalculator import DebyePDFCalculator

#from pdf_lib.glbl import glbl
from glbl import glbl

class PdfLibBuild(object):
    ''' a class to look up cif data and calculate pdf automatically'''

    def __init__(self, API_key, lib_dir = None):
        ''' API_key : user-id-like generated from material project
            lib_dir : directory where you wants to store cif and pdf_data
        '''
        self.API_key = API_key
        test_m = MPRester(API_key)
        print('You are using %s as API key' % API_key)

        ## take care of dir
        if not lib_dir:
            lib_dir = time.strftime('PDF_Lib_%Y-%m-%d')
        working_dir = os.path.expanduser('~/' + lib_dir)
        self.working_dir = working_dir
        # python2
        if os.path.isdir(working_dir):
            pass
        else:
            os.makedirs(working_dir)
        print('Lib dir %s has been built' % working_dir)

    def SpaceGroupLib(self, space_group_symbol, size_limit=None):
        ''' function to build up pdf library based on space group symbol'''
        ## space_group_symbol
        self.space_group_symbol = space_group_symbol
        if isinstance(space_group_symbol, list):
            space_group_symbol_set = space_group_symbol
        else:
            space_group_symbol_set = list(spac_group_symbol)

        ## changing dir
        os.chdir(self.working_dir)
        if os.getcwd() == self.working_dir:
            print('Library will be built at %s' % self.working_dir)
        else:
            print('Werid, return')
            return
        # set up calculation environment
        dbc = DebyePDFCalculator()
        cfg = {'qmin': glbl.q_min, 'qmax':glbl.q_max, 'rmin':glbl.r_min, 'rmax': glbl.r_max}
        Bisoequiv = glbl.Bisoequiv #FIXME: current value = 0.5, need to figure out the most suitable value
        
        for space_group_symbol in space_group_symbol_set:

            print('Building library with space_group symbol: {}'.format(space_group_symbol))
            ## create dirs
            pdf_dir = os.path.join(self.working_dir, space_group_symbol, 'PDF_data')
            cif_dir = os.path.join(self.working_dir, space_group_symbol, 'cif_data')
            if os.path.isdir(pdf_dir):
                pass
            else:
                os.makedirs(pdf_dir)
            if os.path.isdir(cif_dir):
                pass
            else:
                os.makedirs(cif_dir)

            ## search query
            m = MPRester(self.API_key)
            search = m.query(criteria = {"spacegroup.symbol": space_group_symbol},
                            properties = ["material_id"])
            if not search:
                print('Hmm, no reasult. Something wrong')
                print('Stop here.....')
                return
            
            ## crazy looping
            if size_limit:
                dim = 400 # 400 data sets per symbol
            else:
                dim = len(search)
            print('Pull out %s data sets' % dim)
            print('Now, starts to save cif and compute pdf...')
            m_id_list = []  # need it as a reference later
            for i in range(dim):
                ## part 1: grab cif files from data base
                m_id = search[i]['material_id']
                m_id_list.append(m_id)

                m_struc = m.get_structure_by_material_id(m_id)
                m_formula = m_struc.formula
                m_name = m_formula.replace(' ', '')
                
                cif_w = CifWriter(m_struc)
                cif_name = os.path.join(cif_dir, m_name + '.cif')
                cif_w.write_file(cif_name)
                
                if os.path.isfile(cif_name):
                    print('%s.cif has been saved' % str(m_name))
                else:
                    print('Something went wrong at %s cif data' % m_name)

                ## part 2: integrate with diffpy
                struc = loadStructure(cif_name)
                struc.Bisoequiv =  Bisoequiv
                (r,g) = dbc(nosymmetry(struc), **cfg)
         
                gr_name = os.path.join(pdf_dir, m_name +'.gr')
                np.savetxt(gr_name, (r,g))

                if os.path.isfile(gr_name):
                    print('%s.gr has been saved' % str(m_name))
                else:
                    print('Something went wrong on %s gr data' % m_name)
            
            # save final id-symbol list
            m_id_name = os.path.join(self.working_dir, glbl.m_id_list)
            with open(m_id_name, 'w') as f:
                yaml.dump(m_id_list, f)
            if os.path.isfile(m_id_name):
                print('{} has been saved'.format(m_id_name))
            else:
                print('Something went wrong when saving {}'.format(m_id_nam))
