import numpy as np
import torch 

def save_lsm(lsm, ls_num, loss_score, extra=None, loss_fn='HL', in_or_out='input', file_path='outputs/landscape_testiing/'):
    """Helper function to save arrays and tensors as numpy arrays."""
    
    file_name = f'{file_path}ls_{in_or_out}_ls{ls_num}_{loss_fn}_{extra}_{np.round(loss_score, 4)}'
    
    if type(lsm) != type(np.array([])):
        lsm = lsm.detach().cpu().numpy()
        
    np.save(f'{file_name}.npy', lsm)