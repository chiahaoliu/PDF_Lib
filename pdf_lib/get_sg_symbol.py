## get space group symbols with certain criteria
import os
ref = "Space_group_ref.txt"

def get_symbol_list(crystal_system):
    ''' Get short names in one crystal system
    
    argument:
    crystal_system - str - name of crystal system
    '''
    
    import linecache
    
    ind_list = []
    ref_path = os.path.join(os.path.dirname(__file__), ref) # relative path
    if os.path.isfile(ref_path):
        print('open {} as reference'.format(ref_path))
        # FIXME - temporily logic. Refine it later with loding Space_group_ref.txt
        with open(ref_path, 'r') as f:
            for ind, line in enumerate(f):
                if crystal_system in line:
                    ind_list.append(ind-2)
                    # print(ind-2) debug line
        symb_list = []
        for ind in ind_list:
            read = linecache.getline(ref_path, ind)
            position_ind = [ ind for ind, x in enumerate(read) if x== '"']
            #print('position_ind = {}'.format(position_ind)) debug line
            if position_ind:
                head = min(position_ind)
                tail = max(position_ind)
                symb = read[head+1:tail]
                print(symb) #degug line
                symb_list.append(symb)
            else:
                pass
        return symb_list
