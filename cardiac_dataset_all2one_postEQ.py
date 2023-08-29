import numpy as np
from torch.utils.data import Dataset
from utils import *
import transforms_3d
 
class EarlyToLateDatasetWithRVLVTemp(Dataset):
    def __init__(self, npz_name, eq_key = 'eq', cv_split_index=None, norm=False, transform=None, pad=None, thre=False):
        self.transform = transform
        self.img = np.load(npz_name)['img']
        self.loc = np.load(npz_name)['loc']
        if eq_key in list(np.load(npz_name).keys()):
            self.eq = np.load(npz_name)[eq_key]
        self.rv = np.load(npz_name)['rv']
        self.lv = np.load(npz_name)['lv']
        self.myo = np.load(npz_name)['myo']
        self.myo_label = np.load(npz_name)['myo_label']
        self.last_label = np.load(npz_name)['last_label']
        if cv_split_index is not None:
            self.img = self.img[..., cv_split_index]
            self.loc = self.loc[..., cv_split_index]
            self.eq = self.eq[..., cv_split_index]
            self.rv = self.rv[..., cv_split_index]
            self.lv = self.lv[..., cv_split_index]
            self.myo = self.myo[..., cv_split_index]
            self.myo_label = self.myo_label[..., cv_split_index]
            self.last_label = self.last_label[..., cv_split_index]
        self.norm = norm
        self.pad = pad
        self.thre = thre

    def __len__(self):
        post_eq_index = []
        for i,x in enumerate(self.last_label):
            post_eq_index.append(np.arange(int(self.eq[i]),int(x)))

        sum_eq = [0]
        y = 0
        for x in post_eq_index:
            y += len(x)
            sum_eq.append(y)
        return sum_eq[-1]

    def __getitem__(self, idx):
        post_eq_index = []
        for i,x in enumerate(self.last_label):
            post_eq_index.append(np.arange(int(self.eq[i]),int(x)))

        sum_eq = [0]
        y = 0
        for x in post_eq_index:
            y += len(x)
            sum_eq.append(y)

        subject_index = np.where(idx < np.array(sum_eq))[0][0] - 1
        temp_index = post_eq_index[subject_index][idx - sum_eq[subject_index]]
        
        img_ = self.img[...,subject_index]
        loc = self.loc[...,subject_index]
        rv = self.rv[...,subject_index]
        lv = self.lv[..., subject_index]
        myo = self.myo[..., subject_index]

        last_frame = img_[..., -1]
        img = img_[..., temp_index]
        img[img<0] = 0
        last_frame[last_frame<0] = 0
        if self.pad is not None:
            zero_padding = self.pad
            last_frame = np.concatenate((last_frame, zero_padding), axis=2)
            img = np.concatenate((img, zero_padding), axis=2)
        last_frame = last_frame[np.newaxis, ...]
        img = img[np.newaxis, ...]
        if self.norm:
            #amax = np.amax(img_)
            #amin = np.amin(img_)
            #img = img_norm_1(img, amax=amax, amin=amin)[0]
            #last_frame = img_norm_1(last_frame, amax=amax, amin=amin)[0]
            img = img_norm_1(img)[0]
            last_frame = img_norm_1(last_frame)[0]
            amax = np.amax([np.amax(lv), np.amax(rv)])
            amin = np.amin([np.amin(lv), np.amin(rv)])
            lv = img_norm_1(lv, amax=amax, amin=amin)[0]
            rv = img_norm_1(rv, amax=amax, amin=amin)[0]
            myo = img_norm_1(myo, amax=amax, amin=amin)[0]
            
        if self.thre:
            this_rv = rv[temp_index] + 1
            this_lv = lv[temp_index] + 1
            if this_rv >= this_lv * 3:
                #print('here', this_rv/this_lv)
                int_thre = (this_rv) / 5
                img += 1
                img[img>int_thre]=int_thre
                img -= 1

        rv = rv[..., np.newaxis]
        lv = lv[..., np.newaxis]
        #myo = myo[..., np.newaxis]
        rel_temp_index = np.zeros_like(rv)
        rel_temp_index[temp_index,:] = 1
        rel_temp_index[self.eq[subject_index],:] = 1
        #print('rel_temp_index',rel_temp_index)
        curves = np.concatenate((rv, lv, rel_temp_index), axis=1)
        
        
        img = {'image': img, 'center': None, 'angle': None, 'shift': None}
        last_frame = {'image': last_frame, 'center': None, 'angle': None, 'shift': None}
        if self.transform:
            img = self.transform(img)
            last_frame['center'] = img['center']
            last_frame['angle'] = img['angle']
            last_frame['shift'] = img['shift']
            last_frame = self.transform(last_frame)
        return img['image'], last_frame['image'], curves
    
class EarlyToLateDatasetWithRVLVTemp_all(Dataset):
    def __init__(self, npz_name, eq_key = 'eq', cv_split_index=None, norm=False, transform=None, pad=None, thre=False):
        self.transform = transform
        self.img = np.load(npz_name)['img']
        self.loc = np.load(npz_name)['loc']
        if eq_key in list(np.load(npz_name).keys()):
            self.eq = np.load(npz_name)[eq_key]
        self.rv = np.load(npz_name)['rv']
        self.lv = np.load(npz_name)['lv']
        self.myo = np.load(npz_name)['myo']
        self.myo_label = np.load(npz_name)['myo_label']
        self.last_label = np.load(npz_name)['last_label']
        if cv_split_index is not None:
            self.img = self.img[..., cv_split_index]
            self.loc = self.loc[..., cv_split_index]
            self.eq = self.eq[..., cv_split_index]
            self.rv = self.rv[..., cv_split_index]
            self.lv = self.lv[..., cv_split_index]
            self.myo = self.myo[..., cv_split_index]
            self.myo_label = self.myo_label[..., cv_split_index]
            self.last_label = self.last_label[..., cv_split_index]
        self.norm = norm
        self.pad = pad
        self.thre = thre

    def __len__(self):
        post_eq_index = []
        for i,x in enumerate(self.last_label):
            post_eq_index.append(np.arange(int(self.myo_label[i]),int(x)))

        sum_eq = [0]
        y = 0
        for x in post_eq_index:
            y += len(x)
            sum_eq.append(y)
        return sum_eq[-1]

    def __getitem__(self, idx):
        post_eq_index = []
        for i,x in enumerate(self.last_label):
            post_eq_index.append(np.arange(int(self.myo_label[i]),int(x)))

        sum_eq = [0]
        y = 0
        for x in post_eq_index:
            y += len(x)
            sum_eq.append(y)

        subject_index = np.where(idx < np.array(sum_eq))[0][0] - 1
        temp_index = post_eq_index[subject_index][idx - sum_eq[subject_index]]
        
        img_ = self.img[...,subject_index]
        loc = self.loc[...,subject_index]
        rv = self.rv[...,subject_index]
        lv = self.lv[..., subject_index]
        myo = self.myo[..., subject_index]

        last_frame = img_[..., -1]
        img = img_[..., temp_index]
        img[img<0] = 0
        last_frame[last_frame<0] = 0
        if self.pad is not None:
            zero_padding = self.pad
            last_frame = np.concatenate((last_frame, zero_padding), axis=2)
            img = np.concatenate((img, zero_padding), axis=2)
        last_frame = last_frame[np.newaxis, ...]
        img = img[np.newaxis, ...]
        if self.norm:
            #amax = np.amax(img_)
            #amin = np.amin(img_)
            #img = img_norm_1(img, amax=amax, amin=amin)[0]
            #last_frame = img_norm_1(last_frame, amax=amax, amin=amin)[0]
            img = img_norm_1(img)[0]
            last_frame = img_norm_1(last_frame)[0]
            amax = np.amax([np.amax(lv), np.amax(rv)])
            amin = np.amin([np.amin(lv), np.amin(rv)])
            lv = img_norm_1(lv, amax=amax, amin=amin)[0]
            rv = img_norm_1(rv, amax=amax, amin=amin)[0]
            myo = img_norm_1(myo, amax=amax, amin=amin)[0]
            
        if self.thre:
            this_rv = rv[temp_index] + 1
            this_lv = lv[temp_index] + 1
            if this_rv >= this_lv * 3:
                int_thre = (this_rv) / 5
                img += 1
                img[img>int_thre]=int_thre
                img -= 1

        rv = rv[..., np.newaxis]
        lv = lv[..., np.newaxis]
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