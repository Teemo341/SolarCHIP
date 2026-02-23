import os
import pickle
from tqdm import tqdm
from datetime import datetime, timedelta

import torch
import numpy as np
from astropy.io import fits

from global_settings import DATA_ROOT


def read_pt_image(path):
    return torch.load(path, weights_only=True)

def read_fits_image(path):
    return fits.open(path)[1].data

def save_list(dir_list, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dir_list, file)

def load_list(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
# splited by day
def transfer_id_to_date(date_id,y_start=2010, m_start = 5, d_start = 1, y_end=2024, m_end = 7, d_end = 1): # todo
    start_date = datetime(y_start, m_start, d_start)
    end_date = datetime(y_end, m_end, d_end)
    if date_id < 0 or date_id > (end_date - start_date).days*24*60:
        raise ValueError(f'date_id out of range, should be in 0 to {(end_date - start_date).days*24*60}')
    current_date = start_date + timedelta(days=date_id)
    return current_date

def transfer_date_to_id(y_now, m_now, d_now, y_start=2010, m_start = 5, d_start = 1, y_end=2024, m_end = 7, d_end = 1):
    start_date = datetime(y_start, m_start, d_start)
    end_date = datetime(y_end, m_end, d_end)
    current_date = datetime(y_now, m_now, d_now)

    if current_date < start_date or current_date > end_date:
        raise ValueError(f'current_date out of range, should be in {start_date} to {end_date}')
    date_id = (current_date - start_date).days
    date_id *= 24*60 # splited by day, so multiply 24*60 to get the minute id
    return date_id

def get_modal_dir(modal, date_id, y_start=2010, m_start=5, d_start=1, y_end=2024, m_end=12, d_end=31):
    current_date = transfer_id_to_date(date_id, y_start, m_start, d_start, y_end, m_end, d_end)
    # date_str_1 = current_date.strftime('%Y/%m/%d')
    # date_str = current_date.strftime('%Y%m%d')
    if modal == 'hmi':
        path_pt = f"{DATA_ROOT}/hmi/{modal}_pt/{modal}.M_720s.{current_date.year:04d}{current_date.month:02d}{current_date.day:02d}_000000_TAI.pt"
        path_fits = f"{DATA_ROOT}/hmi/{modal}_fits/{modal}.M_720s.{current_date.year:04d}{current_date.month:02d}{current_date.day:02d}_000000_TAI.fits"
    elif modal == '1700': # special because there is no 0000 but 0002
        path_pt = f"{DATA_ROOT}/aia/{modal}_pt/AIA{current_date.year:04d}{current_date.month:02d}{current_date.day:02d}_0002_{modal}.pt"
        path_fits = f"{DATA_ROOT}/aia/{modal}_fits/AIA{current_date.year:04d}{current_date.month:02d}{current_date.day:02d}_0002_{modal}.fits"
    else:
        path_pt = f"{DATA_ROOT}/aia/{modal}_pt/AIA{current_date.year:04d}{current_date.month:02d}{current_date.day:02d}_0000_{modal}.pt"
        path_fits = f"{DATA_ROOT}/aia/{modal}_fits/AIA{current_date.year:04d}{current_date.month:02d}{current_date.day:02d}_0000_{modal}.fits"
    return path_fits, path_pt


def transfer_fits_to_pt(modal, exist_list=None, time_interval = [0,7452000]):
    if not os.path.exists('./data/idx_list'):
        os.makedirs('./data/idx_list')
    if exist_list is None:
        exist_list = np.zeros(time_interval[1], dtype=np.bool)
    else:
        exist_list = load_list(exist_list)
        print(len(exist_list))
    
    move_num = 0
    for i in tqdm(range(time_interval[0], time_interval[1])):
        if exist_list[i] == 0:  # no pt file now
            dir_fits, dir_pt = get_modal_dir(modal, i)
            if os.path.exists(dir_fits):
                try:
                    fits_img = read_fits_image(dir_fits)
                    fits_img = np.nan_to_num(fits_img, nan=0.0)
                    pt_img = torch.tensor(fits_img,dtype=torch.float32)
                    pt_dir = os.path.dirname(dir_pt)
                    if not os.path.exists(pt_dir):
                        os.makedirs(pt_dir)
                    torch.save(pt_img, dir_pt)
                    exist_list[i] = True
                    move_num += 1
                except:
                    pass
        if i % ((time_interval[1]-time_interval[0])//100) == 0:
            save_list(exist_list, f'./data/idx_list/{modal}_exist_idx.pkl')

    print(f'transfer done, {move_num} files transfterd, exist list saved to ./Data/idx_list/{modal}_exist_idx.pkl')


def update_exist_list(modal, save_dir = './data/idx_list', time_interval = [0,5400]): 
    print(f'begin to update {modal} exist list')
    exist_idx = np.zeros(time_interval[1], dtype=np.bool_)
    for i in tqdm(range(time_interval[0], time_interval[1])):
        path_fits, path_pt = get_modal_dir(modal, i)
        if os.path.exists(path_pt):
            exist_idx[i] = True
    save_list(exist_idx, f'{save_dir}/{modal}_exist_idx.pkl')


if __name__ == '__main__':


    modal_list = ['hmi','0094','0131','0171','0193','0211','0304','0335','1600','1700','4500']
    for modal in modal_list:
        update_exist_list(modal)