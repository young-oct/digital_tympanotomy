# -*- coding: utf-8 -*-
# @Time    : 2022-06-07 12:13 p.m.
# @Author  : young wang
# @FileName: digital_tympanotomy.py
# @Software: PyCharm


"""this script performs the following
(1). read .oct file into a numpy array
(2). geometrically correct the distorted view coordinates
(3). export the .oct volume into the DICOM format
(4). save the geometrically correct volume into numpy array for future analysis"""

import glob
from os.path import join
import numpy as np
from scipy.signal import medfilt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time
from functools import partial
from multiprocessing import cpu_count, Pool
from tools.dicom_converter import oct_to_dicom
from tools.auxiliary import folder_creator, arrTolist, listtoarr, load_from_oct_file
from tools.geocrt import getPolarco, iniTri, polar2cart

if __name__ == '__main__':
    #
    oct_files = []

    dst_root_path = '/Users/youngwang/Desktop/tympanotomy/data'

    directory = join(dst_root_path, 'OCT Format')

    path = directory + '/*.oct'
    for filepath in glob.iglob(path):
        oct_files.append(filepath)

    oct_files.sort()
    file_root_name = oct_files[-1].split('/')[-1].split('.')[0]

    raw_data = load_from_oct_file(oct_files[0])
    data = raw_data

    # get polarcords & initialize triangularization
    polarcords = getPolarco(degree=18)
    tri = iniTri(polarcords)

    # construct Cartesian grids
    xq, zq = np.mgrid[0:int(512), 0:int(330)]

    x_list = arrTolist(raw_data, Yflag=False)

    func = partial(polar2cart, tri, xq, zq)
    start = time.time()

    com_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))

    print('starting time:', com_time)

    with Pool(processes=cpu_count()) as p:
        results_list = p.map(func, x_list)

        p.close()
        p.join()

    data_x = listtoarr(results_list, Yflag=False)
    data_xc = np.nan_to_num(data_x).astype(np.uint16)

    y_list = arrTolist(data_xc, Yflag=True)

    with Pool(processes=cpu_count()) as p:
        results_list = p.map(func, y_list)

        p.close()
        p.join()

    data_y = listtoarr(results_list, Yflag=True)
    data = np.nan_to_num(data_y).astype(np.uint16)

    end = time.time()
    print('finished in %.3f seconds' % (end - start))

    # thresold 3D data to remove noise
    temp = np.where(data <= data.max() * 0.65, 0, data)
    img_shape = temp.shape

    # create the mip image from original image
    image_slice = np.amax(temp, axis=2)
    # plt.imshow(image_slice)
    # plt.show()

    # find the centroid of image
    from skimage.measure import moments as moments

    M = moments(image_slice)
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

    x_dir = image_slice[:, int(centroid[0])]
    y_dir = image_slice[int(centroid[1]), :]

    peaks_x, _ = find_peaks(x_dir, )
    peaks_y, _ = find_peaks(y_dir, )

    fig, ax = plt.subplots(1, 5, figsize=(16, 9))

    vmin, vmax = 65, 200

    ax[0].imshow(np.rot90(raw_data[:, 256, :]), cmap='gray', vmin = vmin, vmax = vmax)
    ax[0].set_title('pre-correction', fontsize=18)

    ax[1].imshow(np.rot90(data[:, 256, :]), cmap='gray',vmin = vmin, vmax = vmax)
    ax[1].set_title('post-correction', fontsize=18)

    # axis = 0

    ax[2].scatter(x=peaks_y[0], y=int(centroid[1]), label='point1')
    ax[2].scatter(x=peaks_y[-1], y=int(centroid[1]), label='point2')

    # axis = 1
    ax[2].scatter(x=int(centroid[0]), y=peaks_x[0], label='point3')
    ax[2].scatter(x=int(centroid[0]), y=peaks_x[-1], label='point4')
    ax[2].legend()

    ax[2].imshow(image_slice, cmap='gray',vmin = vmin, vmax = vmax)
    ax[2].scatter(x=centroid[0], y=centroid[1])
    ax[2].set_title('maximum intensity projection', fontsize=18)

    # sample slice prototype

    slice_m = temp[:,256,:]
    peak_loc = np.zeros(int(peaks_x[-1] - peaks_x[0]) + 1)
    peak_wid = np.zeros(int(peaks_x[-1] - peaks_x[0]) + 1)

    for i in range(int(peaks_x[-1] - peaks_x[0]) + 1):
        peaks_locs, _ = find_peaks(slice_m[peaks_x[0] + i, :])
        peak_wid[i] = peaks_locs[-1] - peaks_locs[0]

        peak_loc[i] = peaks_locs[-1]

    for i in range(int(peaks_x[-1] - peaks_x[0]) + 1):
        ax[3].scatter(x=int(peak_loc[i]), y=int(peaks_x[0] + i))

    np.median(peak_wid)
    ax[3].imshow(slice_m, cmap='gray',vmin = vmin, vmax = vmax)
    ax[3].set_title('sample slice prototype', fontsize=18)

    y_range = int(peaks_y[-1] - peaks_y[0] + 1)
    x_range = int(peaks_x[-1] - peaks_x[0] + 1)

    peak_loc_3d = np.zeros((y_range,x_range))
    peak_wid_3d = np.zeros((y_range,x_range))

    for i in range(y_range):
        idx = int(i + peaks_y[0])
        slice_m = temp[:, idx, :]
        for j in range(x_range):
            peaks, _ = find_peaks(slice_m[peaks_x[0] + j, :])

            if len(peaks) != 0:

                peak_loc_3d[i,j] = peaks[-1]
                peak_wid_3d[i,j] = peaks[-1] - peaks[0]
            else:
                pass

    TM_thickness = np.median(peak_wid_3d)

    TM_remove = data
    for i in range(y_range):
        idx = int(i + peaks_y[0])
        slice_m = TM_remove[:, idx, :]
        for j in range(x_range):
            slice_m[int(peaks_x[0] + j),int(peak_loc_3d[i,j]-TM_thickness)::] = 0
        TM_remove[:,idx,:] = slice_m

    ax[4].imshow(np.rot90(TM_remove[:, 256, :]), cmap='gray',vmin = vmin, vmax = vmax)
    ax[4].set_title('TM removal', fontsize=18)

    plt.tight_layout()
    plt.show()

    patient_info = {'PatientName': 'RESEARCH',
                    'PatientBirthDate': '20220707',
                    'PatientSex': 'F',
                    'PatientAge': '0Y',
                    'PatientID': '202207070001',
                    'SeriesDescription': 'Right Ear',
                    'StudyDescription': 'OCT 3D'}

    # temp_path = 'original'
    temp_path = 'TM removal'

    dicom_path = join(dst_root_path, 'DICOM', temp_path)

    folder_creator(dicom_path)

    resolutionx, resolutiony, resolutionz = 0.026, 0.026, 0.030

    oct_to_dicom(TM_remove, resolutionx=resolutionx,
                 resolutiony=resolutiony,resolutionz = resolutionz,
                 dicom_folder=dicom_path,
                 **patient_info)

    print('Done')
