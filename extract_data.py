# -*- coding: utf-8 -*-

import os
import pydicom as dicom
import numpy as np
import SimpleITK as sitk
import fnmatch
from glob import glob
import nibabel as nib
import xlrd
import re
from skimage import transform


# python list sort
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):
    return sorted(v_list, key=str2int)


# 下面的代码是包含了多个切片（slices）的扫描面，我们仅仅是将其存储为python列表。
# 切片的厚度丢失，即Z轴方向上的像素尺寸，但是可以利用其它的值推测出来，加到元数据中。
def load_scan(path):
    # Load the scans in given folder path

    slices = []
    # for s in sorted(os.listdir(path)):
    for s in sort_humanly(os.listdir(path)):
        if fnmatch.fnmatch(s, "*.nii"):
            label = nib.load(os.path.join(path, s))
        elif fnmatch.fnmatch(s, "*[!.nii]"):
            slices.append(dicom.read_file(path + '/' + s))

    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    for slices_number in range(len(slices)):
        intercept = slices[slices_number].RescaleIntercept
        slope = slices[slices_number].RescaleSlope

        if slope != 1:
            image[slices_number] = slope * image[slices_number].astype(np.float64)
            image[slices_number] = image[slices_number].astype(np.int16)

        image[slices_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def read_all_data(data_load, data_save):
    for i in sorted(os.listdir(data_load)):
        for j in sorted(os.listdir(os.path.join(data_load, i))):
            data = os.path.join(data_load, i, j, 'PP')  # if
            data_slice = load_scan(data)
            data_all = get_pixels_hu(data_slice)
            data_all = np.flip(data_all, 0)

            image_array = sitk.GetImageFromArray(data_all)
            # sitk.WriteImage(image_array, data_save + '/{}_{}.nii'.format(i, j))
            sitk.WriteImage(image_array, data_save + '/{}_{}.nii'.format(i, j))


def read_label_info(path, out_dir):
    # 打开excel文件，创建一个workbook对象,表含有sheet名
    rbook = xlrd.open_workbook(path)
    # sheets方法返回对象列表
    rbook.sheets()
    all1 = []
    all2 = []
    for sheet in range(len(rbook.sheets())):
        label1 = []
        label2 = []
        for row in range(rbook.sheet_by_index(sheet).nrows):
            if (rbook.sheet_by_index(sheet).cell(row, 4).value == 'CT') :
                label1.append(rbook.sheet_by_index(sheet).cell(row, 8).value)
                label2.append(rbook.sheet_by_index(sheet).cell(row, 9).value)

        all1.extend(label1)
        all2.extend(label2)

    all1 = np.array(all1)
    all2 = np.array(all2)
    print(all1)
    print(all2.shape)
    print(type(all2))
    # np.savetxt(os.path.join(out_dir, 'label1.csv'), all1)
    # np.savetxt(os.path.join(out_dir, 'label2.csv'), all2)

    return all1, all2


def process_label(data):
    for i in range(data.shape[0]):
        if data[i] == '＋':
            data[i] = 1
        else:
            data[i] = 0
    return data


def process_data(data):
    all = []
    for i in sorted(os.listdir(data)):

        data_infor = nib.load(os.path.join(data, i)).get_data()
        normalization(data_infor)
        all.append(transform.resize(data_infor.astype(np.float32), (64, 64, 32))   # size need to data
)

    return np.array(all)


def normalization(image):

    MIN_BOUND = -100.0
    MAX_BOUND = 400.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image = (image - np.mean(image)) / (np.std(image))

    return image


if __name__ == "__main__":
    path = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/data"
    # path_save = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/data_save"
    path_save = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/data_save_last"
    # read_all_data(path, path_save)

    path_label = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/胃癌-最终2019.12.4.xls"
    label_save = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/label_save"
    # result1, result2 = read_label_info(path_label, label_save)

    # a = process_label(result1)
    # b = process_label(result2)

    # np.save("/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/label1", a.astype(np.uint8))

    result_data = process_data(path_save)

    np.save("/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/data", result_data.astype(np.float16))