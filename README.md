# TAI-GAN
TAI-GAN: Temporally and Anatomically Informed GAN for early-to-late frame conversion in dynamic cardiac PET motion correction. 

To appear in SASHIMI 2023, a MICCAI workshop.

## Dataset

We originally used an internal dataset at Yale PET Center. If you would like to access the data, please contact chi.liu@yale.edu

We encourage users to download and process their own data. 

You will likely need to pre-process your own dataset and save it as an npz file with keys as follows:

`'img'`: the concatenated image volumes with the size of `[*shape, temporal_dim, num_subject]`. Here, for our dataset with 85 subjects, 27-frame dynamic scans and a 3D volume size of [64,64,64], this array shape is [64, 64, 64, 27, 85].

`'rv'`: the time-activity curves (TACs) of right ventricle blood pool (RVBP), with the shape of `[temporal_dim, num_subject]`

`'lv'`: the TACs of left ventricle blood pool (LVBP), with the shape of `[temporal_dim, num_subject]`

`'myo'`: the TACs of myocardium, with the shape of `[temporal_dim, num_subject]`

`'mask'`: the concatenated segmentation volumes, with the same size as `'img'`

`'eq'`: the temporal index of the first frame where the LVBP TAC is equal to or higher than RVBP TAC (Shi et al. IEEE-TMI 2021). This is helpful in stratifying very early frames and for further quantification

`'myo_label'`: the temporal index of the first frame where myocardium activity is higher than 10% maximum. This is the starting point of frame conversion since all the earlier frames don't have a significant effect on MBF quantification and are discarded

`'last_label'`: the temporal index of the last frame for conversion since due to temporal normalization (Shi et al. IEEE-TMI 2021) all the frames later than this index are duplicates and are not sent for conversion
        
If using cross validation, you could specify the subjects for the current split with the training and validation subject indexes.

## Train and evaluation

`python main_cv.py` to run GAN one-to-one (Sundar et al.) and all-to-one baselines.

`python main_cv_film_mask.py` to run the proposed TAI-GAN.

`python main_cv_film.py` and `python main_cv_mask.py` to run the ablation studies.

## Papers

If you use TAI-GAN or some part of the code, please cite:

```
@article{guo2023tai,
  title={TAI-GAN: Temporally and Anatomically Informed GAN for early-to-late frame conversion in dynamic cardiac PET motion correction},
  author={Guo, Xueqi and Shi, Luyao and Chen, Xiongchao and Zhou, Bo and Liu, Qiong and Xie, Huidong and Liu, Yi-Hwa and Palyo, Richard and Miller, Edward J and Sinusas, Albert J and others},
  journal={arXiv preprint arXiv:2308.12443},
  year={2023}
}
```

# Contact
For any problems or questions please open an issue or contact xueqi.guo@yale.edu.  



# Acknowledgments

The code was heavily borrowed from [https://github.com/togheppi/cDCGAN](https://github.com/togheppi/cDCGAN )
