import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime


import sunpy.visualization.colormaps.cm  # noqa: F401  注册 sdoaia/hmimag 等官方色表
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


# ----------------------------------------------------------------------
# 固定显示范围（针对【原始/物理量】数据，保证跨图可比）
#
# 注意：不要用 data/modal_stats.json 的 mean/std —— 那是 log1p+zscore 训练空间
# 的统计，而 solarplot 画的是反归一化后的原始物理量（HMI 高斯 / AIA DN）。
# 这里的范围由原始数据分位数标定（HMI 取实测极值附近 ±2000；AIA 取 p99.9
# 附近，用来压掉宇宙线/坏像素等离群点），所有图共用同一把尺子，
# 图与图之间的亮度/对比度才能互相比较。
#
# HMI 磁图：以 0 为中心对称（发散色系白 = 0 磁场）
# AIA：0 到上限
# ----------------------------------------------------------------------
DISPLAY_LIMITS = {
    'hmi':  (-2000.0, 2000.0),
    '0094': (0.0, 40.0),
    '0131': (0.0, 150.0),
    '0171': (0.0, 2500.0),
    '0193': (0.0, 3000.0),
    '0211': (0.0, 2000.0),
    '0304': (0.0, 500.0),
    '0335': (0.0, 80.0),
    '1600': (0.0, 500.0),
    '1700': (0.0, 2500.0),
    '4500': (0.0, 16000.0),
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


def solarplot(data,
              modal: str,
              time: str,
              save_path: str,
              figsize=(10, 8),
              vmin=None,
              vmax=None):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)

    header = get_header(modal, time)
    mymap = sunpy.map.Map((data, header))

    plt.figure(figsize=figsize)

    # 未显式给出时，使用该模态的固定显示范围（跨图可比）；
    # 也可以手动传 vmin/vmax 覆盖默认值。
    if vmin is None or vmax is None:
        limits = DISPLAY_LIMITS.get(modal)
        if limits is not None:
            vmin = limits[0] if vmin is None else vmin
            vmax = limits[1] if vmax is None else vmax

    if modal == 'hmi':
        # HMI 磁图：hmimag 发散色系 + 固定对称范围（白 = 0 磁场）
        mymap.plot(cmap='hmimag', vmin=vmin, vmax=vmax)
    else:
        # AIA：SunPy 根据 INSTRUME+WAVELNTH 自动匹配 sdoaia 官方色表
        mymap.plot(vmin=vmin, vmax=vmax)

    # plt.colorbar()
    plt.savefig(save_path)
    plt.close()


                