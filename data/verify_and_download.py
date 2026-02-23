import wget
import os 
import numpy as np
import torch
from tqdm import tqdm
import argparse

from data.utils import transfer_id_to_date, get_modal_dir, read_fits_image


def download_convert(modal, time_interval = [0, 1e+32]):
    """
    modal: hmi, 0094, 0131, 0171, 0193, 0211, 0304, 0335, 1600, 1700, 4500
    """
    print('start')
    exist_num = 0
    download_num = 0
    error_url = []
    error_path = f'./data/download_error/{modal}'
    if not os.path.exists(error_path):
        os.mkdir(error_path)

    pbar = tqdm(range(6000))
    for i in pbar:
        if i<time_interval[0] or i>time_interval[1]:
            continue
        
        date_time = transfer_id_to_date(i)
        year = date_time.year
        month = date_time.month
        day = date_time.day
        pbar.set_description(f'{modal} | {year} | {month} | {day}')

        if modal == 'hmi':
            url = f'https://jsoc1.stanford.edu/data/hmi/fits/{year:04d}/{month:02d}/{day:02d}/hmi.M_720s.{year:04d}{month:02d}{day:02d}_000000_TAI.fits'
        elif modal == '1700':
            url = f'https://jsoc1.stanford.edu/data/aia/synoptic/{year:04d}/{month:02d}/{day:02d}/H0000/AIA{year:04d}{month:02d}{day:02d}_0002_{modal}.fits'
        else:
            url = f'https://jsoc1.stanford.edu/data/aia/synoptic/{year:04d}/{month:02d}/{day:02d}/H0000/AIA{year:04d}{month:02d}{day:02d}_0000_{modal}.fits'
        path_fits, path_pt = get_modal_dir(modal, i)

        try:
            dir_fits = os.path.dirname(path_fits)
            if not os.path.exists(dir_fits):
                os.makedirs(dir_fits)
            if os.path.exists(path_fits):
                exist_num += 1
            else:
                wget.download(url, path_fits) # download fits file
                download_num += 1
            try:
                if not os.path.exists(path_pt):
                    fits_img = read_fits_image(path_fits)
                    fits_img = np.nan_to_num(fits_img, nan=0.0)
                    pt_img = torch.tensor(fits_img,dtype=torch.float32)

                    pt_dir = os.path.dirname(path_pt)
                    if not os.path.exists(pt_dir):
                        os.makedirs(pt_dir)
                    torch.save(pt_img, path_pt)
            except Exception as e:
                print(f"Error occured : {e}, delete {path_pt} if exists")
                if os.path.exists(path_pt):
                    os.remove(path_pt)                           
        except Exception as e:
            error_url.append(url)
            with open(f'{error_path}/error_url.txt', 'a') as f:
                f.writelines(f'{error_url[-1]}\n')
        
        # time.sleep(5)
    
    for url in error_url:
        print(url)
    print(f'| {exist_num} already exist, {download_num} success, {len(error_url)} fail, the final try is {modal} modal {year} year {month} month {day} day')
    print('finish')
            
# 2025/02/07 get the exist pickle


if __name__ == '__main__' :

    args = argparse.ArgumentParser()
    args.add_argument('--modal', type=str, default='hmi', help='the modal to download, should be one of hmi, 0094, 0131, 0171, 0193, 0211, 0304, 0335, 1600, 1700, 4500')
    args.add_argument('--start', type=int, default=0, help='the start id to download, should be between 0 and 6000')
    args.add_argument('--end', type=int, default=6000, help='the end id to download, should be between 0 and 6000')
    args = args.parse_args()

    download_convert(args.modal, time_interval=[args.start, args.end])