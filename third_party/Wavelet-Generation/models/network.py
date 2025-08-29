import torch
import torch.nn.functional as F
import numpy as np
import copy
import pywt
from spconv.core import AlgoHint, ConvAlgo
import spconv.pytorch as spconv
from spconv.pytorch.hash import HashTable
from models.module.dwt_utils import prep_filt_sfb3d, prep_filt_afb3d
from models.module.diffusion_network import UNetModel, MyUNetModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(input_size, kernel_size=1, stride=1, pad=0):
    from math import floor, ceil
    h = floor( ((input_size + (2 * pad) - kernel_size )/ stride) + 1)
    return h

def indices_to_key(keys, spatial_size, delta = 50): # HACK
    new_keys = keys[:, 3] + keys[:, 2] * (spatial_size[-1]+delta) + keys[:, 1] * (spatial_size[-1]+delta) * (spatial_size[-2]+delta) + \
    keys[:, 0] * (spatial_size[-1]+delta) * (spatial_size[-2]+delta) * (spatial_size[-3]+delta)

    return new_keys
def create_coordinates(resolution, feature_dim = 1):
    dimensions_samples = np.linspace(0, resolution - 1, resolution)

    if feature_dim > 1:
        feature_samples = np.arange(feature_dim)
        d, x, y, z = np.meshgrid(feature_samples, dimensions_samples, dimensions_samples, dimensions_samples)
        d, x, y, z = np.swapaxes(d[:, :, :, :, np.newaxis], 0, 1),\
                     np.swapaxes(x[:, :, :, :, np.newaxis], 0, 1),\
                     np.swapaxes(y[:, :, :, :, np.newaxis], 0, 1),\
                     np.swapaxes(z[:, :, :, :, np.newaxis], 0, 1)
        coordinates = np.concatenate((d, x, y, z), axis=4)
        coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).cuda(device)
        return coordinates
    else:
        x, y, z = np.meshgrid(dimensions_samples, dimensions_samples, dimensions_samples)
        x, y, z = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis], z[:, :, :, np.newaxis]
        coordinates = np.concatenate((x, y, z), axis=3)
        coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).cuda(device)
        return coordinates

class DummyLayer(torch.nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()
    def forward(self, x):
        return x



class NearestUpsample3D(torch.nn.Module):
    def __init__(self, upsample_ratio):
        super().__init__()
        self.upsample_ratio = upsample_ratio

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upsample_ratio, mode='nearest')
        return x

class DownSampleConv3D(torch.nn.Module):
    def __init__(self, input_dim, output_dim, spatial_size, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        self.activation = self.config.activation

        current_dim = input_dim

        feature_size = spatial_size
        for layer_dim, kernel_size, stride in self.config.conv3d_downsample_tuple_layers:
            layer_list = []

            if stride[0] == 1:
                conv_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=layer_dim,
                                kernel_size=kernel_size, padding='same')
            else:
                conv_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=layer_dim,
                                kernel_size=kernel_size, stride=stride)
            layer_list.append(conv_layer)

            if stride[0] != 1:
                feature_size = conv_output_shape(feature_size, kernel_size[0], stride[0], pad=0)

            if self.config.use_instance_norm:
                norm_layer = torch.nn.InstanceNorm3d(layer_dim, affine=self.config.use_instance_affine)
                layer_list.append(norm_layer)
            if self.config.use_layer_norm:
                norm_layer = torch.nn.LayerNorm([layer_dim, feature_size, feature_size, feature_size],
                                                elementwise_affine=self.config.use_layer_affine)
                layer_list.append(norm_layer)



            new_layer = torch.nn.Sequential(
                *layer_list
            )

            self.layers.append(new_layer)
            current_dim = layer_dim

        for layer in self.layers:
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, 'weight') and hasattr(sublayer, 'bias') and not isinstance(sublayer, torch.nn.InstanceNorm3d) and \
                        not isinstance(sublayer, torch.nn.LayerNorm):
                        torch.nn.init.normal_(sublayer.weight, mean=0.0, std=config.weight_sigma)
                        torch.nn.init.constant_(sublayer.bias, 0)
            else:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
                torch.nn.init.constant_(layer.bias, 0)

        ### last layer
        self.last_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=output_dim, kernel_size=(1,1,1), stride=(1,1,1))

        torch.nn.init.normal_(self.last_layer.weight, mean=0.0, std=config.weight_sigma)
        torch.nn.init.constant_(self.last_layer.bias, 0)


    def forward(self, input_features):

        x = input_features
        batch_size = x.size(0)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        x = self.last_layer(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view((batch_size, -1))

        return x

class Conv3DHigh(torch.nn.Module):
    def __init__(self, input_dim, output_dim, level, config):
        super().__init__()
        self.config = config
        self.desne_layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.activation = self.config.activation

        current_dim = input_dim
        for layer_dim in self.config.conv3d_dense_layers:
            new_layer = torch.nn.Linear(current_dim, layer_dim)
            self.desne_layers.append(new_layer)
            current_dim = layer_dim
        self.desne_layers.append(torch.nn.Linear(current_dim, self.config.conv3d_latent_dim))


        ### conv3d layers
        current_dim = self.config.conv3d_latent_dim // 8

        feature_size = 2
        conv3d_tuple_layers = copy.deepcopy(self.config.conv3d_tuple_layers)
        for i in range(self.config.max_depth - level):
            conv3d_tuple_layers.extend(copy.deepcopy(self.config.conv3d_tuple_layers_highs_append))

        for layer_dim, kernel_size, stride in conv3d_tuple_layers:
            if self.config.conv3d_use_upsample:

                if stride[0] > 1:
                    layer_list = [NearestUpsample3D(stride)]
                else:
                    layer_list = []
                layer_list.append(torch.nn.Conv3d(in_channels=current_dim, out_channels=layer_dim,
                                    kernel_size=kernel_size, padding = 'same'))
                feature_size = int(feature_size * stride[0])
                if self.config.use_instance_norm:
                    norm_layer = torch.nn.InstanceNorm3d(layer_dim, affine=self.config.use_instance_affine)
                    layer_list.append(norm_layer)
                if self.config.use_layer_norm:
                    norm_layer = torch.nn.LayerNorm([layer_dim, feature_size, feature_size, feature_size], elementwise_affine=self.config.use_layer_affine)
                    layer_list.append(norm_layer)

                new_layer = torch.nn.Sequential(
                    *layer_list
                )

            else:
                new_layer = torch.nn.ConvTranspose3d(in_channels=current_dim, out_channels=layer_dim,
                                                     kernel_size=kernel_size,
                                                     stride=stride)
            self.layers.append(new_layer)
            current_dim = layer_dim

        ### last layer
        self.last_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=output_dim, kernel_size=(1,1,1), stride=(1,1,1))

        ### layer initialization
        for layer in self.desne_layers:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
            torch.nn.init.constant_(layer.bias, 0)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, 'weight') and hasattr(sublayer, 'bias') and not isinstance(sublayer, torch.nn.InstanceNorm3d) and \
                        not isinstance(sublayer, torch.nn.LayerNorm):
                        torch.nn.init.normal_(sublayer.weight, mean=0.0, std=config.weight_sigma)
                        torch.nn.init.constant_(sublayer.bias, 0)
            else:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.last_layer.weight, mean=0.0, std=config.weight_sigma)
        torch.nn.init.constant_(self.last_layer.bias, 0)



    def forward(self, codes, spatial_shape):

        ## transform and reshape
        batch_size = codes.size(0)
        x = codes
        for layer in self.desne_layers:
            x = layer(x)
            x = self.activation(x)

        ## re shape
        x = x.view(batch_size, -1, 2, 2, 2)

        ## upsamples
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        ##  last layer
        x = self.last_layer(x)

        low_bound = x.size(2) // 2 - spatial_shape[0] // 2, x.size(3) // 2 - spatial_shape[1] // 2, x.size(4) // 2 - spatial_shape[2] // 2
        delta = spatial_shape[0] % 2, spatial_shape[1] % 2, spatial_shape[2] % 2
        high_bound = x.size(2) // 2 + spatial_shape[0] // 2 + delta[0], x.size(3) // 2 + spatial_shape[1] // 2 + delta[1], x.size(4) // 2 + spatial_shape[2] // 2 + delta[2]
        x = x[:, :, low_bound[0]:high_bound[0], low_bound[1]:high_bound[1], low_bound[2]:high_bound[2]]

        return x

class Conv3D(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.config = config
        self.desne_layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.activation = self.config.activation

        current_dim = input_dim
        for layer_dim in self.config.conv3d_dense_layers:
            new_layer = torch.nn.Linear(current_dim, layer_dim)
            self.desne_layers.append(new_layer)
            current_dim = layer_dim
        self.desne_layers.append(torch.nn.Linear(current_dim, self.config.conv3d_latent_dim))


        ### conv3d layers
        current_dim = self.config.conv3d_latent_dim // 8

        feature_size = 2
        if hasattr(self.config, 'conv3d_tuple_layers'):
            for layer_dim, kernel_size, stride in self.config.conv3d_tuple_layers:
                if hasattr(self.config, 'conv3d_use_upsample') and self.config.conv3d_use_upsample:

                    layer_list = [ NearestUpsample3D(stride),
                        torch.nn.Conv3d(in_channels=current_dim, out_channels=layer_dim,
                                        kernel_size=kernel_size, padding = 'same')]
                    feature_size *= stride[0]
                    if self.config.use_instance_norm:
                        norm_layer = torch.nn.InstanceNorm3d(layer_dim, affine=self.config.use_instance_affine)
                        layer_list.append(norm_layer)
                    if self.config.use_layer_norm:
                        norm_layer = torch.nn.LayerNorm([layer_dim, feature_size, feature_size, feature_size], elementwise_affine=self.config.use_layer_affine)
                        layer_list.append(norm_layer)

                    new_layer = torch.nn.Sequential(
                        *layer_list
                    )

                else:
                    new_layer = torch.nn.ConvTranspose3d(in_channels=current_dim, out_channels=layer_dim,
                                                         kernel_size=kernel_size,
                                                         stride=stride)
                self.layers.append(new_layer)
                current_dim = layer_dim
        else:
            for layer_dim in self.config.conv3d_layers:
                new_layer = torch.nn.ConvTranspose3d(in_channels=current_dim, out_channels=layer_dim, kernel_size=self.config.conv3d_kernel_size,
                                                         stride = (2, 2, 2))
                self.layers.append(new_layer)
                current_dim = layer_dim

        ### last layer
        self.last_layer = torch.nn.Conv3d(in_channels=current_dim, out_channels=output_dim, kernel_size=(1,1,1), stride=(1,1,1))

        ### layer initialization
        for layer in self.desne_layers:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
            torch.nn.init.constant_(layer.bias, 0)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, 'weight') and hasattr(sublayer, 'bias') and not isinstance(sublayer, torch.nn.InstanceNorm3d) and \
                        not isinstance(sublayer, torch.nn.LayerNorm):
                        torch.nn.init.normal_(sublayer.weight, mean=0.0, std=config.weight_sigma)
                        torch.nn.init.constant_(sublayer.bias, 0)
            else:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=config.weight_sigma)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.last_layer.weight, mean=0.0, std=config.weight_sigma)
        torch.nn.init.constant_(self.last_layer.bias, 0)



    def forward(self, codes, spatial_shape):

        ## transform and reshape
        batch_size = codes.size(0)
        x = codes
        for layer in self.desne_layers:
            x = layer(x)
            x = self.activation(x)

        ## re shape
        x = x.view(batch_size, -1, 2, 2, 2)

        ## upsamples
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        ##  last layer
        x = self.last_layer(x)

        low_bound = x.size(2) // 2 - spatial_shape[0] // 2, x.size(3) // 2 - spatial_shape[1] // 2, x.size(4) // 2 - spatial_shape[2] // 2
        delta = spatial_shape[0] % 2, spatial_shape[1] % 2, spatial_shape[2] % 2
        high_bound = x.size(2) // 2 + spatial_shape[0] // 2 + delta[0], x.size(3) // 2 + spatial_shape[1] // 2 + delta[1], x.size(4) // 2 + spatial_shape[2] // 2 + delta[2]
        x = x[:, :, low_bound[0]:high_bound[0], low_bound[1]:high_bound[1], low_bound[2]:high_bound[2]]

        return x

def get_conv_shape(current_spatial_shape, conv_module):
    spatial_shape_out = spconv.ops.get_conv_output_size(current_spatial_shape, kernel_size=conv_module.kernel_size, stride = conv_module.stride, padding=conv_module.padding,
                                    dilation=conv_module.dilation)

    return spatial_shape_out

def get_conv_indices(current_indices, current_spatial_shape, conv_module, batch_size):
    indices_out = spconv.ops.get_indice_pairs(indices=current_indices, batch_size = batch_size, spatial_shape=current_spatial_shape,
                                algo= ConvAlgo.Native, ksize=conv_module.kernel_size, stride=conv_module.stride, padding = conv_module.padding,
                                          dilation=conv_module.dilation, out_padding = conv_module.output_padding)[0]
    spatial_shape_out = spconv.ops.get_conv_output_size(current_spatial_shape, kernel_size=conv_module.kernel_size, stride = conv_module.stride, padding=conv_module.padding,
                                    dilation=conv_module.dilation)
    return indices_out, spatial_shape_out

def compute_modules(conv_dim, input_shape, h0, g0, mode):


    assert mode in ['zero', 'constant']

    N = input_shape[conv_dim]
    L = h0.numel()
    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L

    # padding for input
    input_shape = copy.deepcopy(input_shape)
    if p % 2 == 1:
        input_shape[conv_dim] += 1

    kernel_size = [1, 1, 1]
    kernel_size[conv_dim] = L
    stride = [1, 1, 1]
    stride[conv_dim] = 2
    pad = [0, 0, 0]
    pad[conv_dim] = p // 2
    conv_module = spconv.SparseConv3d(in_channels = 1, out_channels = 1, kernel_size = kernel_size,
                                      stride= stride, padding=pad, bias = False, groups = 1).to(device)
    conv_module.weight = torch.nn.Parameter(torch.reshape(h0 , conv_module.weight.size()).to(device), requires_grad = False)
    pad = [0, 0, 0]
    pad[conv_dim] = L - 2
    inv_module = spconv.SparseConvTranspose3d(in_channels = 1, out_channels = 1, kernel_size = kernel_size,
                                      stride= stride, padding=pad, bias = False, groups = 1).to(device)
    #g0 = torch.flip(g0, dims = [2+conv_dim])
    inv_module.weight = torch.nn.Parameter(torch.reshape(g0 , inv_module.weight.size()).to(device), requires_grad = False)

    output_shape = get_conv_shape(input_shape, conv_module)

    return output_shape, conv_module, inv_module

def initalize_modules(input_shape, max_depth, h0_dep, h0_col, h0_row, g0_dep, g0_col, g0_row, mode):


    ## compute input_indices
    shapes_list = [input_shape]
    current_shape = input_shape
    conv_modules, inv_modules = [], []

    assert mode in ["zero", 'constant']

    ## compute shapes and indices
    for i in range(max_depth):
        current_shape, conv_module_row, inv_module_row = compute_modules(conv_dim = 2, input_shape = current_shape, h0=h0_row, g0 = g0_row, mode = mode)
        current_shape, conv_module_col, inv_module_col = compute_modules(conv_dim = 1, input_shape = current_shape, h0=h0_col, g0 = g0_col, mode = mode)
        current_shape, conv_module_dep, inv_module_dep = compute_modules(conv_dim = 0, input_shape = current_shape, h0=h0_dep, g0 = g0_dep, mode = mode)
        shapes_list.append(current_shape)
        conv_modules.append([conv_module_row, conv_module_col, conv_module_dep])
        inv_modules.append([inv_module_dep, inv_module_col, inv_module_row])

    return shapes_list, conv_modules, inv_modules

class SparseComposer(torch.nn.Module):
    def __init__(self, input_shape, J=1, wave='db1', mode='zero', inverse_dwt_module = None):
        super().__init__()
        self.inverse_dwt_module = inverse_dwt_module
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            h0_dep, h1_dep = h0_col, h1_col


        # Prepare the filters
        filts = prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_dep', filts[0])
        self.register_buffer('h1_dep', filts[1])
        self.register_buffer('h0_col', filts[2])
        self.register_buffer('h1_col', filts[3])
        self.register_buffer('h0_row', filts[4])
        self.register_buffer('h1_row', filts[5])
        self.J = J
        self.mode = mode
        self.input_shape = input_shape

        ## Need for inverse
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col


        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_dep', filts[0])
        self.register_buffer('g1_dep', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])

        ### initalize module
        self.shape_list, self.conv_modules, self.inv_modules = initalize_modules(input_shape = input_shape, max_depth = self.J,
                          h0_dep = self.h0_dep, h0_col = self.h0_col, h0_row = self.h0_row,
                          g0_dep = self.g0_dep, g0_col = self.g0_col, g0_row = self.g0_row,
                          mode = self.mode)


    def forward(self, input_indices, weight_func, **kwargs):

        batch_size, indices_list = self.extract_indcies_list(input_indices)

        ### compute the features from bottom-up
        if self.inverse_dwt_module is None:
            current_coeff = None
            for i in range(self.J)[::-1]:
                kwargs['spatial_shape'] = self.shape_list[i+1]
                output_coeff = weight_func(indices = indices_list[i+1], level = i+1, **kwargs)

                ### add with previous layer
                if current_coeff is not None:
                    current_coeff = current_coeff.unsqueeze(1) + output_coeff
                else:
                    current_coeff = output_coeff

                current_coeff = spconv.SparseConvTensor(features = current_coeff, indices = indices_list[i+1],
                                        spatial_shape = self.shape_list[i+1], batch_size=batch_size)



                ### perform idwf
                current_coeff = self.inv_modules[i][0](current_coeff)
                current_coeff = self.inv_modules[i][1](current_coeff)
                current_coeff = self.inv_modules[i][2](current_coeff)

                ### retrived only useful coeff
                table = HashTable(device, torch.int32, torch.float32, max_size=current_coeff.indices.size(0) * 2)
                coeff_indices, query_indices = indices_to_key(current_coeff.indices, spatial_size=current_coeff.spatial_shape),\
                                               indices_to_key(indices_list[i], spatial_size=current_coeff.spatial_shape)


                table.insert(coeff_indices, current_coeff.features)
                current_coeff, isempty = table.query(query_indices)

                assert sum(isempty) == 0

            kwargs['spatial_shape'] = self.shape_list[0]
            output_coeff = weight_func(indices=indices_list[0], level= 0, **kwargs)
            final_coeff = current_coeff.unsqueeze(1) + output_coeff
        else:
            final_coeff = None
            low, highs = None, []
            for i in range(self.J)[::-1]:
                kwargs['spatial_shape'] = self.shape_list[i+1]
                output_coeff = weight_func(indices = indices_list[i+1], level = i+1, **kwargs)
                current_coeff = spconv.SparseConvTensor(features = output_coeff, indices = indices_list[i+1],
                                        spatial_shape = self.shape_list[i+1], batch_size=batch_size)
                dense_coeff = current_coeff.dense(channels_first = True)
                if i+1 == self.J:
                    low = dense_coeff
                else:
                    highs = [dense_coeff] + highs

            ## last layers
            kwargs['spatial_shape'] = self.shape_list[0]
            output_coeff = weight_func(indices=indices_list[0], level=0, **kwargs)
            current_coeff = spconv.SparseConvTensor(features=output_coeff, indices=indices_list[0],
                                                    spatial_shape=self.shape_list[0], batch_size=batch_size)
            dense_coeff = current_coeff.dense(channels_first=True)
            highs = [dense_coeff] + highs

            final_coeff = self.inverse_dwt_module((low, highs))
            indices_long = indices_list[0].long()
            final_coeff = final_coeff[indices_long[:, 0], 0, indices_long[:, 1], indices_long[:, 2], indices_long[:, 3]].unsqueeze(1)

        return final_coeff

    def extract_indcies_list(self, input_indices):
        ## prepare the indices
        batch_size = input_indices.size(0)
        sample_num = input_indices.size(1)
        batch_indices = torch.arange(0, batch_size).int()
        batch_indices = batch_indices.unsqueeze(1).repeat((1, sample_num)).view((-1, 1)).to(device)
        input_indices = input_indices.view((-1, 3))
        current_indices = torch.cat((batch_indices, input_indices), dim=-1)
        ## compute the indices for each level
        indices_list = [current_indices]
        current_shape = self.input_shape
        for i in range(self.J):
            current_indices, current_shape = get_conv_indices(current_indices=current_indices,
                                                              current_spatial_shape=current_shape,
                                                              conv_module=self.conv_modules[i][0],
                                                              batch_size=batch_size)
            current_indices, current_shape = get_conv_indices(current_indices=current_indices,
                                                              current_spatial_shape=current_shape,
                                                              conv_module=self.conv_modules[i][1],
                                                              batch_size=batch_size)
            current_indices, current_shape = get_conv_indices(current_indices=current_indices,
                                                              current_spatial_shape=current_shape,
                                                              conv_module=self.conv_modules[i][2],
                                                              batch_size=batch_size)
            indices_list.append(current_indices)
        return batch_size, indices_list


if __name__ == "__main__":

    from configs import config
    from models.module.dwt import DWTForward3d, DWTInverse3d


    module = spconv.SparseConvTranspose3d(1, 1, (16, 1, 1), stride = (2, 1, 1), groups= 1, indice_key= 'spconv3').to(device)
    module_2 = spconv.SparseConvTranspose3d(1, 1, (1, 16, 1), stride = (1, 2, 1), groups= 1).to(device)
    module_3 = spconv.SparseConvTranspose3d(1, 1, (1, 1, 16), stride = (1, 1, 2), groups= 1).to(device)
    module.weight = torch.nn.Parameter(torch.zeros_like(module.weight), requires_grad = False)
    #print(module.weight)
    print(module.weight.size())

    resolution = 64

    features =  torch.zeros(resolution * resolution * resolution, 1).to(device)# your features with shape [N, num_channels]
    indices =  create_coordinates(resolution,  1).view(-1, 3).int().to(device)# your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
    indices = torch.cat((torch.zeros((indices.size(0), 1), dtype = torch.int32).to(device),indices), dim = -1)
    spatial_shape =  [resolution, resolution, resolution]# spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
    batch_size = 1 # batch size of your sparse tensor.
    x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

    #print(x)
    #print(indices)
    #print(x_dense_NCHW.size())

    x_out = module(x)
    inverse_module = spconv.SparseInverseConv3d(1, 1, (16, 1, 1), indice_key = 'spconv3').to(device)
    x_out_inverse = inverse_module(x_out)
    print(x_out)
    print(x_out.indices.size())

    table = HashTable(device, torch.int32, torch.float, max_size=x_out.indices.size(0) * 2)
    table.insert(x_out.indices,x_out.features)

    vq, _ = table.query(x_out.indices)
    print(vq)
    print(x_out.spatial_shape)

    indices_conv = spconv.ops.get_indice_pairs(indices=x.indices, batch_size = 1, spatial_shape=x.spatial_shape,
                                algo= ConvAlgo.Native, ksize=module.kernel_size, stride=module.stride, padding = module.padding,
                                          dilation=module.dilation, out_padding = module.output_padding)
    print(indices_conv)
    x_output_size = spconv.ops.get_conv_output_size(x.spatial_shape, kernel_size=module.kernel_size, stride = module.stride, padding=module.padding,
                                    dilation=module.dilation)
    print(x_output_size)





    #dwt_forward_3d = DWTForward3d(J = config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
    #dwt_inverse_3d = DWTInverse3d(J = config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
    #network = MultiScaleMLP(config = config, data_num = 1, dwt_module = dwt_forward_3d, inverse_dwt_module = dwt_inverse_3d).to(device)

    #indices = torch.from_numpy(np.arange(1)).to(device)
    #output = network(indices)
