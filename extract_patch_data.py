# -*- coding: utf-8 -*-

import os
import SimpleITK as sitk
import fnmatch
import nibabel as nib
import re

from tools import get_compact_range


# sort ways
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):
    return sorted(v_list, key=str2int)


def read_mask(path):
    # Load the scans in given folder path
    for label_s in sort_humanly(os.listdir(path)):
        if fnmatch.fnmatch(label_s, "*.nii"):
            label = nib.load(os.path.join(path, label_s)).get_data()

    return label


def read_infor(data_load):

    all_result = []
    for i in sorted(os.listdir(data_load)):
        for j in sorted(os.listdir(os.path.join(data_load, i))):
            data = os.path.join(data_load, i, j, 'PP')
            result = read_mask(data)
        all_result.append(result)
    return all_result


def read_data_info(path):
    all_label = []
    for i in sort_humanly(os.listdir(path)):
        label = nib.load(os.path.join(path, i)).get_data()
        all_label.append(label)

    return all_label


def extract_patch(mask_arr_, img_arr_, data_save):
    # 去掉mask周围的0，得到一个缩小版的mask，再生成image，加速运算
    for i in range(mask_arr_):
        if min(mask_arr_[i].shape) > 5:

            valid_range_z, valid_range_y, valid_range_x = get_compact_range(mask_arr_)

            mask_arr = mask_arr_[
                        valid_range_z[0]: valid_range_z[1] + 1,
                        valid_range_y[0]: valid_range_y[1] + 1,
                        valid_range_x[0]: valid_range_x[1] + 1]

            img_arr = img_arr_[
                        valid_range_z[0]: valid_range_z[1] + 1,
                        valid_range_y[0]: valid_range_y[1] + 1,
                        valid_range_x[0]: valid_range_x[1] + 1]

            img_arr = sitk.GetImageFromArray(img_arr)
            sitk.WriteImage(img_arr, data_save + '/{}.nii'.format(i))

        # for c in range(img_arr.shape[0]):
        #     image_sample = np.array(Image.fromarray(img_arr[c, :, :]).convert("RGB"))
        #     mask_layer = mask_arr.astype(image_sample.dtype)[c]
        #     image_sample[:, :, 0] += mask_layer * 50
        #     image_sample[image_sample > 255] = 255
        #     image_show = Image.fromarray(image_sample)
        #     plt.imshow(image_show)
        #     plt.show()

        # # 保证image和mask的Spacing等信息一致
        # img_itk = sitk.GetImageFromArray(img_arr)
        # img_itk.SetSpacing(img.GetSpacing())
        # img_itk.SetOrigin(img.GetOrigin())
        #
        # mask.SetSpacing(img.GetSpacing())
        # mask.SetOrigin(img.GetOrigin())
        #
        # mask_itk.SetSpacing(img.GetSpacing())
        # mask_itk.SetOrigin(img.GetOrigin())


if __name__ == "__main__":
    path = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/data"
    path_save_ = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/data_save_last"
    path_save = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/03/data_save_last_last"

    data = read_infor(path)
    mask = read_mask(data)
    data = read_data_info(path_save_)
    extract_patch(mask, data, path_save)