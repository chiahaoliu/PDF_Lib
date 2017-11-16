# script to execute database building

import os
from pdf_lib.parallel_func import (save_apply_result, learninglib_build,
                                   map_learninglib)
from ipyparallel import Client

rc = Client()
dview = rc[:]

def run_build(cif_dir):
    fn_list = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
    full_fn_list = list(map(lambda x: os.path.join(cif_dir, x), fn_list))
    rv = dview.apply_async(learninglib_build, full_fn_list)
    save_apply_result(rv)
