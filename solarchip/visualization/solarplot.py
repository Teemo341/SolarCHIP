import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


import sunpy.map
from astropy.io import fits

INSTRUME_DICT = {
    'hmi':  'HMI_SIDE1',
    '0094': 'AIA_4',
    '0131': 'AIA_1',
    '0171': 'AIA_3',
    '0193': 'AIA_2',
    '0211': 'AIA_2',
    '0304': 'AIA_4',
    '0335': 'AIA_1',
    '1600': 'AIA_3',
    '1700': 'AIA_3',
    '4500': 'AIA_3'
}

WAVELNTH_DICT = {
    'hmi':  6173,
    '0094': 94,
    '0131': 131,
    '0171': 171,
    '0193': 193,
    '0211': 211,
    '0304': 304,
    '0335': 335,
    '1600': 1600,
    '1700': 1700,
    '4500': 4500
}

# HMI full-res: ~0.505 arcsec/px @ 4096² → 1024² downsampled ≈ 2.0 arcsec/px
# AIA full-res: ~0.600 arcsec/px @ 4096² → 1024² downsampled = 2.4 arcsec/px
CDELT_DICT = {
    'hmi':  2.0,
    '0094': 2.4,
    '0131': 2.4,
    '0171': 2.4,
    '0193': 2.4,
    '0211': 2.4,
    '0304': 2.4,
    '0335': 2.4,
    '1600': 2.4,
    '1700': 2.4,
    '4500': 2.4
}


def get_header(modal: str,
               time: str):
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 1024
    header['NAXIS2'] = 1024
    header['IMG_TYPE'] = 'LIGHT'

    if modal == 'hmi':
        header['TELESCOP'] = 'SDO/HMI'
        header['OBSRVTRY'] = 'SDO'
    else:
        header['TELESCOP'] = 'SDO/AIA'

    header['DATE-OBS'] =  time       #'2010-06-03T00:00:08.14'
    header['INSTRUME'] = INSTRUME_DICT.get(modal)
    header['WAVELNTH'] = WAVELNTH_DICT.get(modal)
    
    header['WAVEUNIT'] = 'angstrom'

    cdelt = CDELT_DICT.get(modal, 2.4)
    header['CTYPE1'] = 'HPLN-TAN'
    header['CUNIT1'] = 'arcsec'
    header['CRVAL1'] = 0.0
    header['CDELT1'] = cdelt
    header['CRPIX1'] = 512.5
    header['CTYPE2'] = 'HPLT-TAN'
    header['CUNIT2'] = 'arcsec'
    header['CRVAL2'] = 0.0
    header['CDELT2'] = cdelt
    header['CRPIX2'] = 512.5
    header['CROTA2'] = 0.0

    return header

def format_timestamp(time_int: int) -> str:
    # 解析整数时间格式：YYYYMMDDHHMM
    dt = datetime.strptime(str(time_int), '%Y%m%d%H%M')
    
    # 加上固定的秒和毫秒（可根据需要自定义）
    formatted = dt.strftime('%Y-%m-%dT%H:%M:%S') + '.14'
    
    return formatted


def solarplot(data: np.array,
              modal: str,
              time: str,
              save_path: str,
              figsize = (10,8)
              ):
    header = get_header(modal, time)
    mymap = sunpy.map.Map((data, header))

    plt.figure(figsize=figsize)

    if modal == 'hmi':
        # HMI 磁图：正负值对称，使用发散色系
        vmax = np.max(np.abs(data))
        mymap.plot(cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    else:
        # AIA：SunPy 根据 INSTRUME+WAVELNTH 自动匹配 sdoaia 官方色表
        mymap.plot()

    # plt.colorbar()
    plt.savefig(save_path)
    plt.close()


                