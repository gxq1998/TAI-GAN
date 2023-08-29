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
from cardiac_dataset_all2one_postEQ import *

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
num_epochs = 100
eval_interval = 50
save_interval = 50
data_dir = 'RealData_all85_allshifted_medfilt_withEQframes_withRVLVMYO_cropped.npz'
data_eval_dir = 'RealData_all85_allshifted_medfilt_withEQframes_withRVLVMYO_cropped.npz'

device = "cuda:3"
init_model_dir = ''# model_dir
init_epoch = 0
cv_fold = 4 # 0-4
cv_split_npz = '../cardiac_mc/cv_indexes_splits.npz'
train_indexes = np.load(cv_split_npz)['train'][cv_fold]
val_indexes = np.load(cv_split_npz)['valid'][cv_fold]

save_name = 'allmappings_mse_highD_dloss_allto27_lowDlr_randomcroprotateshift_cardiaceval_imgnorm1_avgpool_batch4_{0}/'.format(str(cv_fold))
save_dir = 'cardiac_results/' + save_name
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

# MNIST dataset
transform = transforms.Compose([transforms_3d.RandomCrop3D((64,64,32)), transforms_3d.RandomRotate3D(), transforms_3d.RandomTranslate3D(5)])

cardiac_data = EarlyToLateDatasetWithRVLVTemp_all(data_dir, eq_key = 'curve_labeled_eq', cv_split_index=train_indexes, norm=True, transform=transform) #, pad = np.zeros((128,128,17)))

data_loader = torch.utils.data.DataLoader(dataset=cardiac_data,
                                          batch_size=batch_size,
                                          shuffle=True)

data_loader_for_eval = torch.utils.data.DataLoader(dataset=cardiac_data,
                                          batch_size=1,
                                          shuffle=False)

cardiac_eval_data = EarlyToLateDatasetWithRVLVTemp_all(data_eval_dir, eq_key = 'curve_labeled_eq', cv_split_index=val_indexes, norm=True, pad = np.zeros((64,64,12))) #, transform=transform)

data_eval_loader = torch.utils.data.DataLoader(dataset=cardiac_eval_data,
                                          batch_size=1,
                                          shuffle=False)


img = next(iter(data_loader))[0]
print('img.shape',np.array(img).shape)
inshape = img.shape[2:]
infeats = img.shape[1]

# Models
#G = Generator(G_input_dim, label_dim, num_filters, G_output_dim)
G = Unet(inshape, infeats)
D = Discriminator(D_input_dim, label_dim, num_filters[::-1], D_output_dim)
if len(init_model_dir)>0:
    G.load_state_dict(torch.load(os.path.join(init_model_dir,'generator',str(init_epoch-1))))
    D.load_state_dict(torch.load(os.path.join(init_model_dir,'discriminator',str(init_epoch-1))))

G.to(device)
D.to(device)

# Set the logger
D_log_dir = 'D_logs/' + save_dir
G_log_dir = 'G_logs/' + save_dir
if not os.path.exists(D_log_dir):
    os.makedirs(D_log_dir)
D_writer = SummaryWriter(D_log_dir)

if not os.path.exists(G_log_dir):
    os.makedirs(G_log_dir)
G_writer = SummaryWriter(G_log_dir)

D_eval_log_dir = 'D_eval_logs/' + save_dir
G_eval_log_dir = 'G_eval_logs/' + save_dir
if not os.path.exists(D_eval_log_dir):
    os.makedirs(D_eval_log_dir)
D_eval_writer = SummaryWriter(D_eval_log_dir)

if not os.path.exists(G_eval_log_dir):
    os.makedirs(G_eval_log_dir)
G_eval_writer = SummaryWriter(G_eval_log_dir)


# Loss function
criterion = torch.nn.BCELoss()
mse_loss = nn.MSELoss()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate/4, betas=betas)

# Training GAN
D_avg_losses = []
G_avg_losses = []
D_avg_eval_losses = []
G_avg_eval_losses = []


if init_epoch:
    step = len(cardiac_data)*init_epoch
else:
    step = 0
for epoch in range(init_epoch,num_epochs):
    D_losses = []
    G_losses = []
    D_eval_losses = []
    G_eval_losses = []

    if (epoch+1) % 200 == 0:
        G_optimizer.param_groups[0]['lr'] /= 2
        D_optimizer.param_groups[0]['lr'] /= 2

    # minibatch training
    for i, (img, last_frame, _,_) in enumerate(data_loader):

        # image data
        #print('img shape',img.shape,type(img))
        #print('img',np.amin(np.array(img)),np.amax(np.array(img)))
        mini_batch = img.size()[0]
        x_ = Variable(img.to(device).float())

        # labels
        y_real_ = Variable(torch.ones(mini_batch).to(device))
        y_fake_ = Variable(torch.zeros(mini_batch).to(device))
        last_frame = Variable(last_frame.to(device).float())
        #print('last_frame',last_frame.shape)
        #c_fill_ = Variable(fill[labels].to(device))

        # Train discriminator with real data
        D_real_decision = D(last_frame)
        #print('D_real_decision', D_real_decision.shape)
        D_real_decision = D_real_decision[...,0]
        #print('D_real_decision',D_real_decision.shape)
        D_real_loss = criterion(D_real_decision, y_real_)

        # Train discriminator with fake data
        #z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        #z_ = Variable(z_.to(device))

        #c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
        #c_onehot_ = Variable(onehot[c_].to(device))
        gen_image = G(x_)
        #print('gen_image', np.amin(gen_image.cpu().detach().numpy()), np.amax(gen_image.cpu().detach().numpy()))
        #print('gen_image', gen_image.shape)

        #c_fill_ = Variable(fill[c_].to(device))
        D_fake_decision = D(gen_image)#.squeeze()
        D_fake_decision = D_fake_decision[..., 0]
        D_fake_loss = criterion(D_fake_decision, y_fake_)
        D_mse_loss = mse_loss(gen_image, last_frame)

        # Back propagation
        D_loss = D_real_loss + D_fake_loss + D_mse_loss
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        #z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        #z_ = Variable(z_.to(device))

        #c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
        #c_onehot_ = Variable(onehot[c_].to(device))
        gen_image = G(x_)

        #c_fill_ = Variable(fill[c_].to(device))
        #D_fake_decision = D(gen_image).squeeze()
        D_fake_decision = D(gen_image)  # .squeeze()
        D_fake_decision = D_fake_decision[..., 0]
        G_fake_loss = criterion(D_fake_decision, y_real_)
        G_mse_loss = mse_loss(gen_image, last_frame)
        G_loss = G_fake_loss + G_mse_loss

        # Back propagation
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.data)
        G_losses.append(G_loss.data)

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, num_epochs, i+1, len(data_loader), D_loss.data, G_loss.data))

        # ============ TensorBoard logging ============#
        D_writer.add_scalar('Loss', D_loss.data, step + 1)
        G_writer.add_scalar('Loss', G_loss.data, step + 1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    G.eval()
    D.eval()
    with torch.no_grad():
        for j, (img, last_frame, _,_) in enumerate(data_eval_loader):
            mini_batch = img.size()[0]
            x_ = Variable(img.to(device).float())
            y_real_ = Variable(torch.ones(mini_batch).to(device))
            y_fake_ = Variable(torch.zeros(mini_batch).to(device))
            last_frame = Variable(last_frame.to(device).float())

            D_real_decision = D(last_frame) #
            D_real_decision = D_real_decision[..., 0]
            #D_real_decision = D_real_decision.squeeze()
            #print('D_real_decision, y_real_',D_real_decision.shape, y_real_.shape)
            D_real_loss = criterion(D_real_decision, y_real_)

            gen_image = G(x_)

            D_fake_decision = D(gen_image)  # .squeeze()
            D_fake_decision = D_fake_decision[..., 0]
            #D_fake_decision = D_fake_decision.squeeze()
            D_fake_loss = criterion(D_fake_decision, y_fake_)
            D_eval_loss = D_real_loss + D_fake_loss

            D_fake_decision = D(gen_image)  # .squeeze()
            D_fake_decision = D_fake_decision[..., 0]
            G_eval_loss = criterion(D_fake_decision, y_real_)
            # loss values
            D_eval_losses.append(D_eval_loss.data)
            G_eval_losses.append(G_eval_loss.data)

            D_eval_writer.add_scalar('Loss', D_eval_loss.data, step + 1)
            G_eval_writer.add_scalar('Loss', G_eval_loss.data, step + 1)
            step += 1

        D_avg_eval_loss = torch.mean(torch.FloatTensor(D_eval_losses))
        G_avg_eval_loss = torch.mean(torch.FloatTensor(G_eval_losses))

        # avg loss values for plot
        D_avg_eval_losses.append(D_avg_eval_loss)
        G_avg_eval_losses.append(G_avg_eval_loss)



    G.train()
    D.train()

    if (epoch + 1) % eval_interval == 0:
        plot_loss(D_avg_losses, G_avg_losses, G_avg_losses, epoch, num_epochs, save=True, save_dir=save_dir+'train_')
        plot_loss(D_avg_eval_losses, G_avg_eval_losses, G_avg_eval_losses, epoch, num_epochs, save=True, save_dir=save_dir+'eval_')
        plot_result(G, data_eval_loader, epoch, num_epochs, device, save=True, save_dir=save_dir + 'eval_whole_')
        plot_result(G, data_loader_for_eval, epoch, num_epochs, device, save=True, save_dir=save_dir+'train_')
        calculate_mse_and_ssim(G, data_eval_loader, epoch, num_epochs, device, save=True, save_dir=save_dir + 'eval_whole_ssim_test')
        #save_result(G, data_eval_loader, epoch, device, save=True, save_dir='infer_save_npz_nii/' + save_name + 'eval_whole_')

    if (epoch + 1) % save_interval == 0:
        torch.save(G.state_dict(), os.path.join(model_dir,'generator',str(epoch)))
        torch.save(D.state_dict(), os.path.join(model_dir, 'discriminator', str(epoch)))

# Make gif
loss_plots = []
gen_image_plots = []
for epoch in range(num_epochs):
    if (epoch + 1) % eval_interval == 0:
        # plot for generating gif
        save_fn1 = save_dir + 'eval_cardiac_losses_epoch_{:d}'.format(epoch + 1) + '.png'
        loss_plots.append(imageio.imread(save_fn1))

        save_fn2 = save_dir + 'eval_cardiac_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn2))

imageio.mimsave(save_dir + 'eval_cardiac_losses_epochs_{:d}'.format(num_epochs) + '.gif', loss_plots, fps=1)
imageio.mimsave(save_dir + 'eval_cardiac_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=1)