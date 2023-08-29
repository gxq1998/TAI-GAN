import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import ndimage
import nibabel as nib
#from sklearn.metrics import mean_squared_error as mse

# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x, device):
    if torch.cuda.is_available():
        x = x.to(device)
    return Variable(x)

def mse(y_true, y_pred):
    output_errors = np.average((y_true - y_pred) ** 2)
    return output_errors

def nmae(y_true, y_pred):
    output_errors = np.average(abs(y_true - y_pred) / (np.amax(y_true) - np.amin(y_true)))
    return output_errors

# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def img_norm_1(image_series, amax=None, amin=None):
    if amax is None:
        amax = np.amax(image_series)
    if amin is None:
        amin = np.amin(image_series)
    image_series = (image_series - amin - 1e-5) / (amax - amin + 1e-5)
    image_series = (image_series - 0.5) * 2
    return image_series, amax, amin

def img_norm(image_series, thres=0, if_otsu=False, normalizer='mean'):
    """normalize image series"""
    num_time_pts = image_series.shape[0]
    img_normalizer = 1.0

    # normalization for each frame using the maximum
    for i in range(num_time_pts):
        if normalizer == 'mean':
            img_normalizer = np.mean(image_series[i,...])
        elif normalizer == 'max':
            img_normalizer = np.max(image_series[i,...])
        if np.mean(image_series[i,...]) != 0:
            image_series[i,...] = image_series[i,...] / img_normalizer
            img_temp = image_series[i,...]
            img_flatten = np.ndarray.flatten(img_temp)
            img_thres1 = np.percentile(img_flatten, thres)
            img_thres = img_thres1
            if if_otsu:
                img_flatten = img_flatten[img_flatten > img_thres1]
                img_thres2 = threshold_otsu(img_flatten)
                img_thres = img_thres2
            img_temp[img_temp <= img_thres] = 0
            image_series[i,...] = img_temp

    return image_series

# Plot losses
def plot_loss(d_losses, g_losses, g_sim_losses, num_epoch, num_epochs, save=False, save_dir='MNIST_cDCGAN_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.plot(g_sim_losses, label='Generator Similarity')
    plt.legend()

    # save figure
    if save:
        save_fn = save_dir + 'cardiac_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, data_eval_loader, num_epoch, num_epochs, device, save=False, save_dir='MNIST_cDCGAN_results/', show=False, fig_size=(3, 12)):
    generator.eval()

    n_rows = 17
    n_cols = 3
    vis_idx = 20
    iterer = iter(data_eval_loader)
    ssim_list = []
    mse_list = []
    img_list = []
    last_list = []
    gen_list = []
    num_of_figs = int(len(data_eval_loader) / (n_rows)) 
    with torch.no_grad():
        for j in range(num_of_figs):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
            for i,ax in enumerate(axes.flatten()):
                if i%3 == 0:
                    next_iterer = next(iterer)
                    img = next_iterer[0]
                    last_frame = next_iterer[1]
                    if len(next_iterer)>2:
                        v = next_iterer[2]
                        v = Variable(v.to(device).float())
                    #print('i',i)
                    print('eval last_frame',np.amin(np.array(last_frame)),np.amax(np.array(last_frame)))
                    x_ = Variable(img[:,0:1,...].to(device).float())
                    #x_ = Variable(img.to(device).float())
                    #gen_image = generator(x_,v)
                    gen_image = generator(x_)
                    gen_img_tmp = gen_image.cpu().data.squeeze().numpy()
                    last_frame_tmp = last_frame.cpu().data.squeeze().numpy()
                    print('eval gen_image', np.amin(gen_img_tmp), np.amax(gen_img_tmp))
                    print('ssim',ssim(gen_img_tmp,last_frame_tmp))
                    print('mse',mse(gen_img_tmp,last_frame_tmp), i)
                    ssim_list.append(ssim(gen_img_tmp,last_frame_tmp))
                    mse_list.append(mse(gen_img_tmp,last_frame_tmp))
                    gen_image = gen_image.cpu().data[...,vis_idx].squeeze()
                    last_frame = last_frame[..., vis_idx].squeeze()
                    img = img[...,vis_idx].squeeze()
                    img_list.append(img.numpy())
                    last_list.append(last_frame.numpy())
                    gen_list.append(gen_image.numpy())
                    #img[img < 0] = 0
                    #img[img > 50000] = 50000
                    #last_frame[last_frame < 0] = 0
                    #last_frame[last_frame > 50000] = 50000
                    #gen_image[gen_image < 0] = 0
                    #gen_image[gen_image > 50000] = 50000
                #print(np.amin(img),np.amax(img))
                ax.axis('off')
                ax.set_adjustable('box')
                if i%3 == 0:
                    if len(img.shape)>2:
                        ax.imshow(img[0,:,:], cmap='gray', aspect='equal')
                        mask_slice = img[1,:,:]
                        #print('np.unique(mask_slice)',np.unique(mask_slice))
                        mask_slice = np.ma.masked_where(abs(mask_slice - -1) < 6e-1, mask_slice)
                        ax.imshow(mask_slice, cmap='autumn', aspect='equal',vmin=-1,vmax=1)
                    else:
                        ax.imshow(img, cmap='gray', aspect='equal')
                elif i%3 == 1:
                    ax.imshow(last_frame, cmap='gray', aspect='equal')
                    if len(img.shape)>2:
                        ax.imshow(mask_slice, cmap='autumn', aspect='equal')
                else:
                    ax.imshow(gen_image.numpy(), cmap='gray', aspect='equal') #cmap='gray',
                    if len(img.shape)>2:
                        ax.imshow(mask_slice, cmap='autumn', aspect='equal')
            plt.subplots_adjust(wspace=0, hspace=0)
            title = 'Epoch {0} Avg SSIM {1:.4f} +- {2:.4f} MSE {3:.4f} +- {4:.4f}'.format(num_epoch+1, np.mean(ssim_list), np.std(ssim_list), np.mean(mse_list), np.std(mse_list))
            fig.text(0.5, 0.04, title, ha='center')
            # save figure
            if save:
                save_fn = save_dir + 'cardiac_epoch_{:d}_'.format(num_epoch+1) + str(j) + '_{:d}'.format(vis_idx) + '.png'
                plt.savefig(save_fn)

    generator.train()

    if show:
        plt.show()
    else:
        plt.close()
        
def calculate_mse_and_ssim(generator, data_eval_loader, num_epoch, num_epochs, device, save=True, save_dir='MNIST_cDCGAN_results/'):
    generator.eval()
    iterer = iter(data_eval_loader)
    ssim_list = []
    mse_list = []
    nmae_list = []
    psnr_list = []

    with torch.no_grad():
        for i in range(len(data_eval_loader)):
            next_iterer = next(iterer)
            img = next_iterer[0]
            last_frame = next_iterer[1]
            if len(next_iterer)>2:
                v = next_iterer[2]
                v = Variable(v.to(device).float())
            #print('eval last_frame',np.amin(np.array(last_frame)),np.amax(np.array(last_frame)))
            #print('cal i',i)
            x_ = Variable(img[:,0:1,...].to(device).float())
            #x_ = Variable(img.to(device).float())
            #gen_image = generator(x_,v)
            gen_image = generator(x_)
            gen_img_tmp = gen_image.cpu().data.squeeze().numpy()
            last_frame_tmp = last_frame.cpu().data.squeeze().numpy()
            #print('eval gen_image', np.amin(gen_img_tmp), np.amax(gen_img_tmp))
            #print('ssim',ssim(gen_img_tmp,last_frame_tmp))
            #print('mse',mse(gen_img_tmp,last_frame_tmp))
            ssim_list.append(ssim(last_frame_tmp,gen_img_tmp,data_range=2))
            mse_list.append(mse(last_frame_tmp,gen_img_tmp))
            nmae_list.append(nmae(last_frame_tmp,gen_img_tmp))
            psnr_list.append(psnr(last_frame_tmp,gen_img_tmp,data_range=2))

    generator.train()
    
    print('Epoch {0} Avg SSIM {1:.4f} +- {2:.4f} MSE {3:.4f} +- {4:.4f} NMAE {5:.4f} +- {6:.4f}'.format(num_epoch+1, np.mean(ssim_list), np.std(ssim_list), np.mean(mse_list), np.std(mse_list), np.mean(nmae_list), np.std(nmae_list)))

    # save lists
    if save:
        save_fn = save_dir + 'epoch_{:d}'.format(num_epoch+1) + '.npz'
        np.savez(save_fn, ssim_list=ssim_list, mse_list=mse_list, nmae_list=nmae_list, psnr_list=psnr_list)
        
def save_result(generator, data_eval_loader, num_epoch, device, save=False, save_dir='MNIST_cDCGAN_results/', show=False, fig_size=(3, 12)):
    
    num_of_subjects = 1
    generator.eval()
    iterer = iter(data_eval_loader)
    img = next(iter(data_eval_loader))[0]
    [subject_index,temp_index] = next(iter(data_eval_loader))[-1]
    inshape = img.shape[2:]
    image_array = np.zeros_like(data_eval_loader.dataset.img)
    #zero_padding = np.zeros((64,64,12,27,num_of_subjects))
    #image_array = np.concatenate((image_array,zero_padding),axis=2)
    gen_array = np.zeros_like(data_eval_loader.dataset.img)
    #gen_array = np.concatenate((gen_array,zero_padding),axis=2)
    #cv_split_index = data_eval_loader.dataset.cv_split_index
    with torch.no_grad():
        for i in range(len(iterer)):
            next_iterer = next(iterer)
            img = next_iterer[0]
            last_frame = next_iterer[1]
            print('np.amax(img)',np.amax(np.array(img)))
            if len(next_iterer)>3:
                v = next_iterer[2][:,:27,:3]
                v = Variable(v.to(device).float())
                subject_index = next_iterer[3][0]
                temp_index = next_iterer[3][1]
            else:
                subject_index = next_iterer[2][0]
                temp_index = next_iterer[2][1]
            x_ = Variable(img.to(device).float())
            #x_ = Variable(img[:,0:1,...].to(device).float())
            gen_image = generator(x_,v)
            #gen_image = generator(x_)
            gen_image = gen_image.cpu().data.squeeze()
            print(subject_index,temp_index,'np.amax(gen_image)',np.amax(np.array(gen_image)))
            img = img.squeeze()
            last_frame = last_frame.squeeze()
            #print('img.shape',img.shape)
            image_array[...,temp_index,subject_index]=img[0,:,:,:]
            gen_array[...,temp_index,subject_index]=gen_image#[:,:,:36]
            image_array[...,-1,subject_index]=last_frame#[:,:,:36]

    # save figure
    if save:
        for i in range(num_of_subjects):
            img = image_array[...,i]
            gen = gen_array[...,i]
            temp_mask = np.unique(np.where(img==0)[-1])
            temp_mask = temp_mask[temp_mask>10]
            print('temp_mask',temp_mask)
            for j in temp_mask:
                img[...,j] = img[...,-1]
                image_array[...,j,i] = img[...,-1]
                gen[...,j] = gen[...,-1]
                gen_array[...,j,i] = gen[...,-1]
            img_affine = np.array([[-1., 0., 0., 63.5], [0., 1., 0., -63.5], [0., 0., 1., -23.], [0., 0., 0., 1.]])
            array_img = nib.Nifti1Image(img, img_affine)
            img_MC_name = save_dir + 'epoch_{:d}'.format(num_epoch + 1) + '_{0}_ori.nii'.format(i)
            nib.save(array_img, img_MC_name)
            array_img = nib.Nifti1Image(gen, img_affine)
            img_MC_name = save_dir + 'epoch_{:d}'.format(num_epoch + 1) + '_{0}_gen.nii'.format(i)
            nib.save(array_img, img_MC_name)
        save_np = save_dir + 'epoch_nii_{:d}'.format(num_epoch + 1) + '.npz'
        np.savez(save_np, img=image_array, gen_image=gen_array)
        print('finished.')
        
def save_result_single(generator, data_eval_loader, num_epoch, device, save=False, save_dir='MNIST_cDCGAN_results/', show=False, fig_size=(3, 12)):
    
    generator.eval()
    iterer = iter(data_eval_loader)
    img = next(iter(data_eval_loader))[0]
    [subject_index,temp_index] = next(iter(data_eval_loader))[-1]
    inshape = img.shape[2:]
    image_array = []
    gen_array = []
    last_array = []
    with torch.no_grad():
        for i in range(len(iterer)):
            next_iterer = next(iterer)
            img = next_iterer[0]
            last_frame = next_iterer[1]
            
            if len(next_iterer)>3:
                v = next_iterer[2]
                v = Variable(v.to(device).float())
                subject_index = next_iterer[3][0]
                temp_index = next_iterer[3][1]
            else:
                subject_index = next_iterer[2][0]
                temp_index = next_iterer[2][1]
            x_ = Variable(img.to(device).float())
            #x_ = Variable(img[:,0:1,...].to(device).float())
            #gen_image = generator(x_,v)
            gen_image = generator(x_)
            gen_image = gen_image.cpu().data.squeeze()
            img = img.squeeze()
            last_frame = last_frame.squeeze()
            #print('img.shape',img.shape)
            image_array.append(img)
            gen_array.append(gen_image)
            last_array.append(last_frame)

    # save figure
    if save:
        save_np = save_dir + 'epoch_{:d}'.format(num_epoch + 1) + '.npz'
        np.savez(save_np, img=np.array(image_array), gen=np.array(gen_array), last=np.array(last_array))
        
def add_init_variability(img, crop_coord, pos_shift_init_vec):
    """add initial motion variability to an image frame"""
    # only return image

    x_low = crop_coord[0] + pos_shift_init_vec[0]
    x_high = crop_coord[1] + pos_shift_init_vec[0]
    y_low = crop_coord[2] + pos_shift_init_vec[1]
    y_high = crop_coord[3] + pos_shift_init_vec[1]
    z_low = crop_coord[4] + pos_shift_init_vec[2]
    z_high = crop_coord[5] + pos_shift_init_vec[2]

    xl = np.linspace(x_low, x_high - 1, 64)
    yl = np.linspace(y_low, y_high - 1, 64)
    zl = np.linspace(z_low, z_high - 1, 36)
    xx, yy, zz = np.meshgrid(xl, yl, zl, indexing='ij')
    img = ndimage.map_coordinates(img, [xx, yy, zz], order=1)
    return img

def series_crop(img, loc, num_time_pts, image_size, window_size, pos_shift_zero_vec):
    # img.shape = [*image_size, num_time_pts]
    loc_x = loc[0]
    loc_y = loc[1]
    loc_z = loc[2]
    image_series = np.zeros((window_size[0], window_size[1], window_size[2], num_time_pts, 2), dtype='float32')

    """determine initial cropping range"""
    x_low = int(
        -loc_x + image_size[0] / 2 - window_size[0] / 2)  # z-axis is inverted compared with Amide coord
    x_high = x_low + window_size[0]
    if x_low < 0:
        x_low = 0
        x_high = x_low + window_size[0]
    elif x_high > image_size[0]:
        x_high = image_size[0]
        x_low = image_size[0] - window_size[0]

    y_low = int(loc_y + image_size[1] / 2 - window_size[1] / 2)
    y_high = y_low + window_size[1]
    if y_low < 0:
        y_low = 0
        y_high = y_low + window_size[1]
    elif y_high > image_size[1]:
        y_high = image_size[1]
        y_low = image_size[1] - window_size[1]

    z_low = int(-loc_z + image_size[2] / 2 - window_size[
        2] / 2 + 1)  # z-axis is inverted compared with Amide coord
    z_high = z_low + window_size[2]
    if z_low < 0:
        z_low = 0
        z_high = z_low + window_size[2]
    elif z_high > image_size[2]:
        z_high = image_size[2]
        z_low = image_size[2] - window_size[2]
    crop_coord = [x_low, x_high, y_low, y_high, z_low, z_high]

    for i in range(num_time_pts):
        image_frame = img[..., i]  # before cropping window
        image_frame_last = img[..., -1]  # before cropping window
        image_series[..., i, 1] = add_init_variability(image_frame_last, crop_coord, pos_shift_zero_vec)
        image_series[..., i, 0] = add_init_variability(image_frame, crop_coord, pos_shift_zero_vec)

    return image_series