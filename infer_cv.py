# MNIST image generation using Conditional DCGAN: https://github.com/togheppi/cDCGAN
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import transforms_3d
import os
import imageio
from torch.utils.tensorboard import SummaryWriter
from networks import *
from utils import *
from cardiac_dataset_one2one_mask import *

# Parameters
image_size = 32
label_dim = 10
G_input_dim = 100
G_output_dim = 1
D_input_dim = 1
D_output_dim = 1
num_filters = [256, 128, 64]#[128, 64, 32]
thre = False

learning_rate = 0.0002
betas = (0.5, 0.999) #MNIST (0.5, 0.999)
batch_size = 4
num_epochs = 200
eval_interval = 50
save_interval = 50
data_eval_dir = 'RealData_all85_allshifted_medfilt_withEQframes_withRVLVMYO_cropped.npz'

device = "cuda:3"
init_model_dir = ''# model_dir
init_epoch = 0
cv_fold = 1 # 0-4
cv_split_npz = '../cardiac_mc/cv_indexes_splits.npz'
train_indexes = np.load(cv_split_npz)['train'][cv_fold]
val_indexes = np.load(cv_split_npz)['valid'][cv_fold]

save_name = 'allmappings_mse_highD_masknorm_trans5_dloss_allto27_lowDlr_randomcroprotateshift_cardiaceval_imgnorm1_avgpool_batch4_{0}/'.format(str(cv_fold))
save_dir = 'cardiac_results/infer/' + save_name
model_dir = 'models/' + save_name

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(os.path.join(model_dir,'generator')):
    os.makedirs(os.path.join(model_dir,'generator'))

if not os.path.exists(os.path.join(model_dir,'discriminator')):
    os.makedirs(os.path.join(model_dir,'discriminator'))

use_cuda = torch.cuda.is_available()
device = torch.device(device if use_cuda else "cpu")
print(torch.cuda.get_device_name(1))
torch.backends.cudnn.benchmark = True

cardiac_eval_data = EarlyToLateDatasetWithRVLVTemp(data_eval_dir, eq_key = 'curve_labeled_eq', cv_split_index=val_indexes, norm=True, pad = np.zeros((64,64,12)))

data_eval_loader = torch.utils.data.DataLoader(dataset=cardiac_eval_data,
                                          batch_size=1,
                                          shuffle=False)

img, last_frame, _,_ = next(iter(data_eval_loader))
inshape = img.shape[2:]
infeats = img.shape[1]

# Models
G = Unet(inshape, infeats)
D = Discriminator(D_input_dim, label_dim, num_filters[::-1], D_output_dim)
if len(init_model_dir)>0:
    G.load_state_dict(torch.load(os.path.join(init_model_dir,'generator',str(init_epoch-1))))
    D.load_state_dict(torch.load(os.path.join(init_model_dir,'discriminator',str(init_epoch-1))))

G.to(device)
D.to(device)


if init_epoch:
    step = len(cardiac_data)*init_epoch
else:
    step = 0
    
for epoch in range(init_epoch,num_epochs):
    
    G.eval()
    D.eval()

    if (epoch + 1) % eval_interval == 0:
        G.load_state_dict(torch.load(os.path.join(model_dir,'generator',str(epoch))))
        D.load_state_dict(torch.load(os.path.join(model_dir,'discriminator',str(epoch))))
        #plot_result(G, data_eval_loader, epoch, num_epochs, device, save=True, save_dir=save_dir + 'all_on_eq-1_eval_whole_ssim_test')
        calculate_mse_and_ssim(G, data_eval_loader, epoch, num_epochs, device, save=True, save_dir=save_dir + 'all_on_eq-1_whole_ssim_test')
        #save_result_single(G, data_eval_loader, epoch, device, save=True, save_dir=save_dir + 'all_on_eq+1_eval_whole_ssim_test')

