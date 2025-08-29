import os
import importlib
import torch
import numpy as np
import mcubes
import torch.nn.functional as F
from models.network import SparseComposer
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from models.module.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, SpacedDiffusion, space_timesteps
from utils.debugger import MyDebugger
from models.module.diffusion_network import UNetModel, MyUNetModel
import time
def process_state_dict(network_state_dict):
    for key, item in list(network_state_dict.items()):
        if 'module.' in key:
            new_key = key.replace('module.', '')
            network_state_dict[new_key] = item
            del network_state_dict[key]

    return network_state_dict




## setting for testing
config_folder = os.path.join('configs')
config_path = os.path.join(config_folder, 'config.py')
spec = importlib.util.spec_from_file_location('*', config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

high_level_config_path = os.path.join(config_folder, 'config_highs.py')
spec = importlib.util.spec_from_file_location('*', high_level_config_path)
high_level_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(high_level_config)

testing_cnt = 201 ## testing cnt
clip_noise = False
use_ddim = False
ddim_eta = 1.0
respacing = [config.diffusion_step // 10]


### Setting of Diffusion Models
diffusion_folder = r'pretrain/pretrained/airplane/' # put the category
epoch = 3980 # diffusion model epoch number
test_index = config.training_stage
network_path = os.path.join(diffusion_folder, f'model_epoch_{test_index}_{epoch}.pth')

### setting for detail predictor
high_level_folder = r'pretrain/pretrained/airplane/highs' # put the category
high_level_epoch = 240 # detail predictor model epoch number
high_test_index = high_level_config.training_stage
high_level_network_path =  os.path.join(high_level_folder, f'model_epoch_{high_test_index}_{high_level_epoch}.pth')




### debugger
from configs import config as current_config





def one_generation_process(args):

    cuda_id, start_index, testing_cnt, folder_path = args
    device = torch.device(f'cuda:{cuda_id}')

    ### level_indices_remap
    level_map = {0 : 3, 1 : 2}

    with torch.no_grad():
        ### initialize network
        dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(
            device)
        dwt_forward_3d_lap = DWTForward3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(
            device)
        composer_parms = dwt_inverse_3d_lap if config.use_dense_conv else None
        dwt_sparse_composer = SparseComposer(input_shape=[config.resolution, config.resolution, config.resolution],
                                             J=config.max_depth,
                                             wave=config.wavelet, mode=config.padding_mode,
                                             inverse_dwt_module=composer_parms).to(
            device)
        network = UNetModel(in_channels=1,
                            model_channels=config.unet_model_channels,
                            out_channels=2 if hasattr(config,
                                                      'diffusion_learn_sigma') and config.diffusion_learn_sigma else 1,
                            num_res_blocks=config.unet_num_res_blocks,
                            channel_mult=config.unet_channel_mult_low,
                            attention_resolutions=config.attention_resolutions,
                            dropout=0,
                            dims=3,
                            activation=config.unet_activation if hasattr(config, 'unet_activation') else None)

        network_state_dict = torch.load(network_path,map_location=f'cuda:{cuda_id}')
        network_state_dict = process_state_dict(network_state_dict)

        network.load_state_dict(network_state_dict)
        network = network.to(device)
        network.eval()

        high_level_network = MyUNetModel(in_channels= 1,
                            spatial_size= dwt_sparse_composer.shape_list[high_level_config.training_stage][0],
                            model_channels=high_level_config.unet_model_channels,
                            out_channels= 1,
                            num_res_blocks=high_level_config.unet_num_res_blocks,
                            channel_mult=high_level_config.unet_channel_mult,
                            attention_resolutions=high_level_config.attention_resolutions,
                            dropout=0,
                            dims=3)
        high_level_network_state_dict = torch.load(high_level_network_path, map_location=f'cuda:{cuda_id}')
        high_level_network_state_dict = process_state_dict(high_level_network_state_dict)
        high_level_network.load_state_dict(high_level_network_state_dict)
        high_level_network = high_level_network.to(device)
        high_level_network.eval()


        betas = get_named_beta_schedule(config.diffusion_beta_schedule, config.diffusion_step,
                                        config.diffusion_scale_ratio)

        diffusion_module = SpacedDiffusion(use_timesteps=space_timesteps(config.diffusion_step, respacing),
                                           betas=betas,
                                           model_var_type=config.diffusion_model_var_type,
                                           model_mean_type=config.diffusion_model_mean_type,
                                           loss_type=config.diffusion_loss_type)

        testing_indices = [265] * testing_cnt
        noise = None


        for m in range(testing_cnt):
            testing_sample_index = testing_indices[m]

            low_lap = torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[config.max_depth])).float().to(
                device)
            highs_lap = [torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[j])).float().to(device) \
                         for j in range(config.max_depth)]

            model_kwargs = {'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt')}
            if use_ddim:
                low_samples = diffusion_module.ddim_sample_loop(model=network,
                                                                shape=[1, 1] + dwt_sparse_composer.shape_list[-1],
                                                                device=device,
                                                                clip_denoised=clip_noise, progress=True,
                                                                noise=noise,
                                                                eta=ddim_eta,
                                                                model_kwargs=model_kwargs).detach()
            else:
                low_samples = diffusion_module.p_sample_loop(model=network,
                                                             shape=[1, 1] + dwt_sparse_composer.shape_list[-1],
                                                             device=device,
                                                             clip_denoised=clip_noise, progress=True, noise=noise,
                                                             model_kwargs=model_kwargs).detach()

            highs_samples = [torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[i]), device=device) for i in
                             range(config.max_depth)]

            upsampled_low = F.interpolate(low_samples, size=tuple(dwt_sparse_composer.shape_list[high_test_index]))
            highs_samples[high_test_index] = high_level_network(upsampled_low)


            voxels_pred = dwt_inverse_3d_lap((low_samples, highs_samples))
            vertices, traingles = mcubes.marching_cubes(voxels_pred.detach().cpu().numpy()[0, 0], 0.0)
            vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
            mcubes.export_obj(vertices, traingles, os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}.obj'))

            print(f"Done {os.path.join(folder_path,f'{m+start_index}_{testing_sample_index}.off')}!")

if __name__ == '__main__':

    debugger = MyDebugger(f'Network-Marching-Cubes-Diffusion-Gen',
                          is_save_print_to_file=False)

    from torch.multiprocessing import Pool

    GPU_CNT = 1
    PER_GPU_PROCESS = 1
    pool = Pool(GPU_CNT * PER_GPU_PROCESS)


    args = []
    assert testing_cnt % (GPU_CNT * PER_GPU_PROCESS) == 0
    if GPU_CNT * PER_GPU_PROCESS > 1:
        per_process_data_num = testing_cnt // (GPU_CNT * PER_GPU_PROCESS)
        for i in range(GPU_CNT):
            for j in range(PER_GPU_PROCESS):
                args.append((i, (i * PER_GPU_PROCESS + j) * per_process_data_num, per_process_data_num, debugger.file_path('.')))

        pool.map(one_generation_process, args)
    else:
        one_generation_process((0, 0, testing_cnt,  debugger.file_path('.')))

    print("done!")

