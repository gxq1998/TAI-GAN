import numpy as np
from torch.utils.data import Dataset
from utils import *

class EarlyToLateDataset(Dataset):
    def __init__(self, npz_name, eq_key = 'eq', cv_split_index=None, crop_size=None, norm=False, transform=None, pad=None):
        self.transform = transform
        self.img = np.load(npz_name)['img']
        self.loc = np.load(npz_name)['loc']
        self.eq = None
        if eq_key in list(np.load(npz_name).keys()):
            self.eq = np.load(npz_name)[eq_key]
        if cv_split_index is not None:
            self.img = self.img[..., cv_split_index]
            self.loc = self.loc[..., cv_split_index]
        if self.eq is not None:
            self.eq = self.eq[..., cv_split_index]
        self.crop_size = crop_size
        self.norm = norm
        self.pad = pad

    def __len__(self):
        return self.img.shape[-1]

    def __getitem__(self, idx):
        # luyao: 128, 128, 47, 27, 65
        
        # one-to-one mappings
        subject_index = idx
        temp_index = self.eq[subject_index]-1 # 5

        img = self.img[...,subject_index]
        loc = self.loc[...,subject_index]
        num_time_pts = img.shape[-1]
        if self.crop_size is not None:
            img = series_crop(img, loc, num_time_pts, image_size=img.shape[:-1],
                              window_size=self.crop_size, pos_shift_zero_vec = [0, 0, 0])
            img = img[..., 0]
        last_frame = img[..., -1]
        img = img[..., temp_index]
              
        img[img<0] = 0
        last_frame[last_frame<0] = 0
        if self.pad is not None:
            zero_padding = self.pad
            last_frame = np.concatenate((last_frame, zero_padding), axis=2)
            img = np.concatenate((img, zero_padding), axis=2)
        last_frame = last_frame[np.newaxis, ...]
        img = img[np.newaxis, ...]
        if self.norm:
            img = img_norm_1(img)[0]
            last_frame = img_norm_1(last_frame)[0]
        img = {'image': img, 'center': None, 'angle': None}
        last_frame = {'image': last_frame, 'center': None, 'angle': 'None'}
        if self.transform:
            img = self.transform(img)
            last_frame['center'] = img['center']
            last_frame['angle'] = img['angle']
            last_frame = self.transform(last_frame)
        return img['image'], last_frame['image']


class EarlyToLateDatasetWithRVLV(Dataset):
    def __init__(self, npz_name, eq_key = 'eq', cv_split_index=None, norm=False, transform=None, pad=None):
        self.transform = transform
        self.img = np.load(npz_name)['img']
        self.loc = np.load(npz_name)['loc']
        if eq_key in list(np.load(npz_name).keys()):
            self.eq = np.load(npz_name)[eq_key]
        self.rv = np.load(npz_name)['rv']
        self.lv = np.load(npz_name)['lv']
        if cv_split_index is not None:
            self.img = self.img[..., cv_split_index]
            self.loc = self.loc[..., cv_split_index]
            self.eq = self.eq[..., cv_split_index]
            self.rv = self.rv[..., cv_split_index]
            self.lv = self.lv[..., cv_split_index]
        self.norm = norm
        self.pad = pad

    def __len__(self):
        return self.img.shape[-1]#*(self.img.shape[-2]-1)

    def __getitem__(self, idx):
        # one-to-one mappings
        subject_index = idx
        temp_index = self.eq[subject_index]-1

        img = self.img[...,subject_index]
        loc = self.loc[...,subject_index]
        rv = self.rv[...,subject_index]
        lv = self.lv[..., subject_index]

        last_frame = img[..., -1]
        img = img[..., temp_index]
        img[img<0] = 0
        last_frame[last_frame<0] = 0
        if self.pad is not None:
            zero_padding = self.pad
            last_frame = np.concatenate((last_frame, zero_padding), axis=2)
            img = np.concatenate((img, zero_padding), axis=2)
        last_frame = last_frame[np.newaxis, ...]
        img = img[np.newaxis, ...]
        if self.norm:
            img = img_norm_1(img)[0]
            last_frame = img_norm_1(last_frame)[0]
            amax = np.amax([np.amax(lv), np.amax(rv)])
            amin = np.amin([np.amin(lv), np.amin(rv)])
            lv = img_norm_1(lv, amax=amax, amin=amin)[0]
            rv = img_norm_1(rv, amax=amax, amin=amin)[0]

        rv = rv[np.newaxis, ...]
        lv = lv[np.newaxis, ...]
        curves = np.concatenate((rv, lv), axis=0)
        img = {'image': img, 'center': None, 'angle': None}
        last_frame = {'image': last_frame, 'center': None, 'angle': 'None'}
        if self.transform:
            img = self.transform(img)
            last_frame['center'] = img['center']
            last_frame['angle'] = img['angle']
            last_frame = self.transform(last_frame)
        return img['image'], last_frame['image'], curves
    
class EarlyToLateDatasetWithRVLVTemp(Dataset):
    def __init__(self, npz_name, eq_key = 'eq', cv_split_index=None, norm=False, transform=None, pad=None):
        self.transform = transform
        self.img = np.load(npz_name)['img']
        self.loc = np.load(npz_name)['loc']
        if eq_key in list(np.load(npz_name).keys()):
            self.eq = np.load(npz_name)[eq_key]
        self.rv = np.load(npz_name)['rv']
        self.lv = np.load(npz_name)['lv']
        self.myo = np.load(npz_name)['myo']
        self.myo_label = np.load(npz_name)['myo_label']
        if cv_split_index is not None:
            self.img = self.img[..., cv_split_index]
            self.loc = self.loc[..., cv_split_index]
            self.eq = self.eq[..., cv_split_index]
            self.rv = self.rv[..., cv_split_index]
            self.lv = self.lv[..., cv_split_index]
            self.myo = self.myo[..., cv_split_index]
            self.myo_label = self.myo_label[..., cv_split_index]
        self.norm = norm
        self.pad = pad

    def __len__(self):
        return self.img.shape[-1]

    def __getitem__(self, idx):
        subject_index = idx
        temp_index = self.eq[subject_index] - 1
        
        img = self.img[...,subject_index]
        loc = self.loc[...,subject_index]
        rv = self.rv[...,subject_index]
        lv = self.lv[..., subject_index]
        myo = self.myo[..., subject_index]

        last_frame = img[..., -1]
        img = img[..., temp_index]
        img[img<0] = 0
        last_frame[last_frame<0] = 0
        if self.pad is not None:
            zero_padding = self.pad
            last_frame = np.concatenate((last_frame, zero_padding), axis=2)
            img = np.concatenate((img, zero_padding), axis=2)
        last_frame = last_frame[np.newaxis, ...]
        img = img[np.newaxis, ...]
        if self.norm:
            img = img_norm_1(img)[0]
            last_frame = img_norm_1(last_frame)[0]
            amax = np.amax([np.amax(lv), np.amax(rv)])
            amin = np.amin([np.amin(lv), np.amin(rv)])
            lv = img_norm_1(lv, amax=amax, amin=amin)[0]
            rv = img_norm_1(rv, amax=amax, amin=amin)[0]
            myo = img_norm_1(myo, amax=amax, amin=amin)[0]

        rv = rv[..., np.newaxis]
        lv = lv[..., np.newaxis]
        myo = myo[..., np.newaxis]
        rel_temp_index = np.zeros_like(rv)
        rel_temp_index[temp_index,:] = 1
        rel_temp_index[self.eq[subject_index],:] = 1
        curves = np.concatenate((rv, lv, rel_temp_index), axis=1)
        
        
        img = {'image': img, 'center': None, 'angle': None, 'shift': None}
        last_frame = {'image': last_frame, 'center': None, 'angle': None, 'shift': None}
        if self.transform:
            img = self.transform(img)
            last_frame['center'] = img['center']
            last_frame['angle'] = img['angle']
            last_frame['shift'] = img['shift']
            last_frame = self.transform(last_frame)

        return img['image'], last_frame['image'], curves, [subject_index,temp_index]
    
class EarlyToLateDatasetWithTempCode(Dataset):
    def __init__(self, npz_name, eq_key = 'eq', cv_split_index=None, norm=False, transform=None, pad=None):
        self.transform = transform
        self.img = np.load(npz_name)['img']
        self.loc = np.load(npz_name)['loc']
        if eq_key in list(np.load(npz_name).keys()):
            self.eq = np.load(npz_name)[eq_key]
        self.rv = np.load(npz_name)['rv']
        self.lv = np.load(npz_name)['lv']
        self.myo = np.load(npz_name)['myo']
        self.myo_label = np.load(npz_name)['myo_label']
        if cv_split_index is not None:
            self.img = self.img[..., cv_split_index]
            self.loc = self.loc[..., cv_split_index]
            self.eq = self.eq[..., cv_split_index]
            self.rv = self.rv[..., cv_split_index]
            self.lv = self.lv[..., cv_split_index]
            self.myo = self.myo[..., cv_split_index]
            self.myo_label = self.myo_label[..., cv_split_index]
        self.norm = norm
        self.pad = pad

    def __len__(self):
        return self.img.shape[-1]#*(self.img.shape[-2]-1)

    def __getitem__(self, idx):
        # one-to-one mappings
        subject_index = idx
        temp_index = self.eq[subject_index]-1

        img = self.img[...,subject_index]
        loc = self.loc[...,subject_index]
        rv = self.rv[...,subject_index]
        lv = self.lv[..., subject_index]
        myo = self.myo[..., subject_index]

        last_frame = img[..., -1]
        img = img[..., temp_index]
        img[img<0] = 0
        last_frame[last_frame<0] = 0
        if self.pad is not None:
            zero_padding = self.pad
            last_frame = np.concatenate((last_frame, zero_padding), axis=2)
            img = np.concatenate((img, zero_padding), axis=2)
        last_frame = last_frame[np.newaxis, ...]
        img = img[np.newaxis, ...]
        if self.norm:
            img = img_norm_1(img)[0]
            last_frame = img_norm_1(last_frame)[0]
            amax = np.amax([np.amax(lv), np.amax(rv)])
            amin = np.amin([np.amin(lv), np.amin(rv)])
            lv = img_norm_1(lv, amax=amax, amin=amin)[0]
            rv = img_norm_1(rv, amax=amax, amin=amin)[0]
            myo = img_norm_1(myo, amax=amax, amin=amin)[0]

        img = {'image': img, 'center': None, 'angle': None}
        last_frame = {'image': last_frame, 'center': None, 'angle': 'None'}
        if self.transform:
            img = self.transform(img)
            last_frame['center'] = img['center']
            last_frame['angle'] = img['angle']
            last_frame = self.transform(last_frame)
        this_rv = rv[temp_index]
        this_lv = lv[temp_index]
        this_myo = myo[temp_index]
        rel_temp_index = (temp_index - self.eq[subject_index])
        temp_code = np.array([this_rv, this_lv, this_myo, rel_temp_index])
        
        return img['image'], last_frame['image'], temp_code