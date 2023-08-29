import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FiLM3D(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return (gammas * x) + betas

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 1]  # decoder
    ]
    return nb_features

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        BN = getattr(nn, 'BatchNorm%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.bn = BN(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        #out = self.bn(out)
        out = self.activation(out)
        return out

class FinalConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        BN = getattr(nn, 'BatchNorm%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.bn = BN(out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.main(x)
        #out = self.bn(out)
        out = self.activation(out)
        return out

class DeConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'ConvTranspose%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)
        #self.activation = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class UnetWithCurveFiLM(torch.nn.Module):
    """
    A unet architecture with FiLM
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 curve_length=27,
                 curve_channel=3,
                 film_dim=32):

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.film_dim = film_dim

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'AvgPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path) with FiLM
        prev_nf = infeats
        
        encoder_nfs = [prev_nf]
        self.encoder = nn.Sequential()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.Sequential()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf
        
        # configure FiLM generator

        self.film = FiLM3D()

        self.film_lstm = nn.LSTM(curve_channel, film_dim, batch_first=True)
        self.film_conv = nn.Conv1d(curve_length, 1, kernel_size=3, stride=1, padding=1)
        self.film_linear = nn.Linear(film_dim, 2*film_dim)
        
        # Initializer
        torch.nn.init.normal(self.film_linear.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(self.film_linear.bias, 0.0)
        
        torch.nn.init.normal(self.film_conv.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(self.film_conv.bias, 0.0)



    def forward(self, x, v):
        # FiLM generator

        film_params, (_,_) = self.film_lstm(v)
        film_params = self.film_conv(film_params)
        film_params = torch.squeeze(film_params, dim=1)
        film_params = self.film_linear(film_params)
        gamma = film_params[..., :self.film_dim]
        beta = film_params[..., self.film_dim:]

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            #film_params = self.encoder_film[level](v)
            #film_dim = int(film_params.shape[-1]/2)
            #gamma = film_params[..., :film_dim]
            #beta = film_params[..., film_dim:]
            #x = self.film(x, gamma, beta)
            x_history.append(x)
            x = self.pooling[level](x)

        x = self.film(x, gamma, beta)
        
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            #film_params = self.decoder_film[level](v)
            #film_dim = int(film_params.shape[-1]/2)
            #gamma = film_params[..., :film_dim]
            #beta = film_params[..., film_dim:]
            #x = self.film(x, gamma, beta)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x_ = x_history.pop()
                #x_ = self.film(x_, gamma, beta)
                x = torch.cat([x, x_], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x
    
class UnetWithFiLM(torch.nn.Module):
    """
    A unet architecture with FiLM
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 curve_length=3,
                 curve_channel=2,
                 film_dim=32):

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.film_dim = film_dim

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'AvgPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path) with FiLM
        prev_nf = infeats
        
        encoder_nfs = [prev_nf]
        self.encoder = nn.Sequential()
        #self.encoder_film = nn.Sequential()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            #encoder_linears = nn.Sequential()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                #encoder_linears.append(nn.Linear(curve_length, 2*nf))
                #print('encoder convs',convs)
                prev_nf = nf
            self.encoder.append(convs)
            #self.encoder_film.append(encoder_linears)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.Sequential()
        #self.decoder_film = nn.Sequential()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            #decoder_linears = nn.Sequential()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                #decoder_linears.append(nn.Linear(curve_length, 2*nf))
                #print('decoder convs',convs)
                prev_nf = nf
            self.decoder.append(convs)
            #self.decoder_film.append(decoder_linears)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            #if num == len(final_convs)-1:
            #    self.remaining.append(FinalConvBlock(ndims, prev_nf, nf))
            #else:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf
        
        # configure FiLM generator
        '''
        self.film_gamma_generator = nn.Sequential()
        input_gamma_linear = nn.Linear(curve_length, film_dim)
        self.film_gamma_generator.add_module('input_gamma_linear', input_gamma_linear)
        self.film_beta_generator = nn.Sequential()
        input_beta_linear = nn.Linear(curve_length, film_dim)
        self.film_beta_generator.add_module('input_beta_linear', input_beta_linear)
        self.film_pooling = nn.AdaptiveAvgPool1d(film_dim)
        '''
        self.film = FiLM3D()
        
        self.film_generator = nn.Sequential()
        film_linear = nn.Linear(curve_length, 2*film_dim)
        self.film_generator.add_module('film_linear', film_linear)
        
        #film_conv = nn.Conv1d(curve_channel, 2*film_dim, kernel_size=3, stride=1, padding=1)
        #self.film_generator.add_module('film_conv', film_conv)
        '''
        film_linear_1 = nn.Linear(curve_length, 16)
        self.film_generator.add_module('film_linear_1', film_linear_1)
        
        film_act = nn.LeakyReLU()
        self.film_generator.add_module('film_act_1', film_act)
        
        film_linear_2 = nn.Linear(16, 32)
        self.film_generator.add_module('film_linear_2', film_linear_2)
        self.film_generator.add_module('film_act_2', film_act)
        
        film_linear_3 = nn.Linear(32, 2*film_dim)
        self.film_generator.add_module('film_linear_3', film_linear_3)
        '''
        
        
        # Initializer
        torch.nn.init.normal(film_linear.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(film_linear.bias, 0.0)
        '''
        torch.nn.init.normal(film_linear_1.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(film_linear_1.bias, 0.0)
        torch.nn.init.normal(film_linear_2.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(film_linear_2.bias, 0.0)
        torch.nn.init.normal(film_linear_3.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(film_linear_3.bias, 0.0)
        '''


    def forward(self, x, v):
        # FiLM generator
        '''
        gamma = self.film_gamma_generator(v)
        gamma = gamma[..., 0]
        gamma = self.film_pooling(gamma)
        beta = self.film_beta_generator(v)
        beta = beta[..., 0]
        beta = self.film_pooling(beta)
        '''
        film_params = self.film_generator(v)
        gamma = film_params[..., :self.film_dim]
        beta = film_params[..., self.film_dim:]

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            #film_params = self.encoder_film[level](v)
            #film_dim = int(film_params.shape[-1]/2)
            #gamma = film_params[..., :film_dim]
            #beta = film_params[..., film_dim:]
            #x = self.film(x, gamma, beta)
            x_history.append(x)
            x = self.pooling[level](x)

        x = self.film(x, gamma, beta)
        
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            #film_params = self.decoder_film[level](v)
            #film_dim = int(film_params.shape[-1]/2)
            #gamma = film_params[..., :film_dim]
            #beta = film_params[..., film_dim:]
            #x = self.film(x, gamma, beta)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x_ = x_history.pop()
                #x_ = self.film(x_, gamma, beta)
                x = torch.cat([x, x_], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x
    
    
class Unet(torch.nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'AvgPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                #print('convs',convs)
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            #if num == len(final_convs)-1:
            #    self.remaining.append(FinalConvBlock(ndims, prev_nf, nf))
            #else:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x_ = x_history.pop()
                #print('decoder x.shape', x.shape, 'x_history.pop().shape', x_.shape)
                x = torch.cat([x, x_], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


# Generator model
class Generator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                # For input
                input_deconv = torch.nn.ConvTranspose2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)

                # Initializer
                torch.nn.init.normal(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(input_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer1.add_module('input_bn', torch.nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.ReLU())

                # For label
                label_deconv = torch.nn.ConvTranspose2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer2.add_module('label_deconv', label_deconv)

                # Initializer
                torch.nn.init.normal(label_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(label_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer2.add_module('label_bn', torch.nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.ReLU())
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Initializer
                torch.nn.init.normal(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(deconv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        self.output_dim = output_dim
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = torch.nn.Conv3d(input_dim, int(num_filters[i]), kernel_size=4, stride=2, padding=1)
                self.hidden_layer1.add_module('input_conv', input_conv)

                # Initializer
                torch.nn.init.normal(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(input_conv.bias, 0.0)

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.LeakyReLU(0.2))

                # For label
                label_conv = torch.nn.Conv3d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer2.add_module('label_conv', label_conv)

                # Initializer
                torch.nn.init.normal(label_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(label_conv.bias, 0.0)

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                conv = torch.nn.Conv3d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                conv_name = 'conv' + str(i + 1)
                self.hidden_layer.add_module(conv_name, conv)

                # Initializer
                torch.nn.init.normal(conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(conv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm3d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv3d(num_filters[i], output_dim, kernel_size=4, stride=4, padding=0)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)

        # Output linear
        self.output_linear_layer = torch.nn.Sequential()
        out2 = torch.nn.Linear(4, output_dim) #[64, 64, 32] 4
        self.output_linear_layer.add_module('out2', out2)
        # Initializer
        torch.nn.init.normal(out2.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out2.bias, 0.0)

        # Activation
        self.output_linear_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, z):
        #print(z.shape)
        h1 = self.hidden_layer1(z)
        #print('h1.shape',h1.shape)
        #h2 = self.hidden_layer2(c)
        #x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(h1)
        #print('h.shape',h.shape)
        out = self.output_layer(h)
        out = torch.reshape(out,(out.shape[0],-1))
        #print('out.shape', out.shape)
        out = self.output_linear_layer(out)
        #print('out.shape', out.shape)
        return out
