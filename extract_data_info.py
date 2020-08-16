# -*- coding: utf-8 -*-

'''
获取数据
'''

import os
import pydicom as dicom
import numpy as np
import SimpleITK as sitk
import fnmatch

import nibabel as nib
import xlrd
import re
from skimage import transform


# sort
def _tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def _str2int(v_str):
    return [_tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def _sort_humanly(v_list):
    return sorted(v_list, key=_str2int)


def get_compact_range(mask_arr):

    z, x, y = mask_arr.shape[0], mask_arr.shape[2], mask_arr.shape[1]
    i_sum = []
    j_sum = []
    k_sum = []
    for i in range(z):
        if np.sum(mask_arr[i, :, :]) > 0:
            i_sum.append(i)

    for j in range(x):
        if np.sum(mask_arr[:, j, :]) > 0:
            j_sum.append(j)

    for k in range(y):
        if np.sum(mask_arr[:, :, k]) > 0:
            k_sum.append(k)

    a = []
    a.append(i_sum[0])
    a.append(i_sum[-1])
    b = []
    b.append(j_sum[0])
    b.append(j_sum[-1])
    c = []
    c.append(k_sum[0])
    c.append(k_sum[-1])
    # print(np.array(a), np.array(b), np.array(c))
    return np.array(a), np.array(b), np.array(c)


# 下面的代码是包含了多个切片（slices）的扫描面，我们仅仅是将其存储为python列表。
# 切片的厚度丢失，即Z轴方向上的像素尺寸，但是可以利用其它的值推测出来，加到元数据中。
def load_scan(path):
    # Load the scans in given folder path
    slices = []
    # for s in sorted(os.listdir(path)):
    for s in _sort_humanly(os.listdir(path)):
        if fnmatch.fnmatch(s, "*.nii"):
            label = nib.load(os.path.join(path, s)).get_data()
        elif fnmatch.fnmatch(s, "*[!.nii]"):
            slices.append(dicom.read_file(path + '/' + s))

    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices, label


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
            data_slice, mask = load_scan(data)
            data_all = get_pixels_hu(data_slice)
            data_all = np.flip(data_all, 0)  # Reverse the order of elements in an array along the given axis.
            mask = np.transpose(mask, (2, 0, 1))
            data_extract = extract_patch(mask, data_all)
            image_array = sitk.GetImageFromArray(data_extract)
            sitk.WriteImage(image_array, data_save + '/{}_{}.nii'.format(i, j))


def extract_patch(mask_arr, img_arr):
    # 去掉mask周围的0，得到一个缩小版的mask，再生成image，加速运算
    if min(mask_arr.shape) > 5:
        valid_range_z, valid_range_y, valid_range_x = get_compact_range(mask_arr)
        print(valid_range_z, valid_range_y, valid_range_x)
        img_arr = img_arr[
                    valid_range_z[0]: valid_range_z[1] + 1,
                    valid_range_y[0]: valid_range_y[1] + 1,
                    valid_range_x[0]: valid_range_x[1] + 1]

    return np.array(img_arr)


# read csv_label information
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
            if (rbook.sheet_by_index(sheet).cell(row, 4).value == 'CT'):
                label1.append(rbook.sheet_by_index(sheet).cell(row, 8).value)
                label2.append(rbook.sheet_by_index(sheet).cell(row, 9).value)

        all1.extend(label1)
        all2.extend(label2)

    all1 = np.array(all1)
    all2 = np.array(all2)

    label_one = process_label(all1)
    label_two = process_label(all2)

    # print(os.path.join(out_dir, label1))
    np.save("/media/03/label1", label_one.astype(np.uint8))
    np.save("/media/03/label2", label_two.astype(np.uint8))

    return label_one, label_two


def process_label(label_value):
    for i in range(label_value.shape[0]):
        if label_value[i] == '＋':
            label_value[i] = 1
        else:
            label_value[i] = 0
    return label_value


# data resize
def process_data(data):
    all = []
    for i in sorted(os.listdir(data)):
        data_infor = nib.load(os.path.join(data, i)).get_data()
        data_infor = normalization(data_infor)
        all.append(transform.resize(data_infor.astype(np.float32), (64, 64, 16)))  # size need to data
    all_data = np.array(all)
    # save array data
    np.save("/media/03/data", all_data.astype(np.float16))
    return all_data


def normalization(image):
    # hu value
    MIN_BOUND = -100.0
    MAX_BOUND = 400.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image = (image - np.mean(image)) / (np.std(image))

    return image


if __name__ == "__main__":
    raw_data = "/media/03/data"
    save_data = "/media/03/data_save_last_"

    raw_label = "/media/03/cancer.xls"
    save_label = "/media//03/label_save"

    data_save_new = "/media/03/data_save_new"

    process_data(save_data)
