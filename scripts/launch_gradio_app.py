"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import functools

import torch
import torch as th
from omegaconf import OmegaConf

from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.util import fix_seed
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from layout_diffusion.dataset.data_loader import build_loaders
from scripts.get_gradio_demo import get_demo
from layout_diffusion.dataset.util import image_unnormalize_batch
import numpy as np
from peft import PeftModel, PeftConfig

import hashlib

object_name_to_idx = {'inhalation-hazard': 0, 'poison': 1, 'flammable': 2, 'radioactive': 3, 'oxidizer': 4, 'explosive': 5, 'corrosive': 6, 'flammable-solid': 7, 'spontaneously-combustible': 8, 'oxygen': 9, 'dangerous': 10, 'infectious-substance': 11, 'marine-toxicity': 12, 'organic-peroxide': 13, 'miscellaneous-materials': 14, 'batteries': 15, 'non-flammable-gas': 16, '__image__': 17, '__null__': 18}

# Function to log incompatible keys
def log_incompatible_keys(missing_keys, unexpected_keys, size_mismatched_keys):
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        for key in missing_keys:
            print(f"  - {key}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        for key in unexpected_keys:
            print(f"  - {key}")
    if size_mismatched_keys:
        print(f"Size mismatched keys: {len(size_mismatched_keys)}")
        for key, expected_shape, loaded_shape in size_mismatched_keys:
            print(f"  - {key}: expected {expected_shape}, but got {loaded_shape}")

def preprocess_state_dict(state_dict):
    new_state_dict = {}

    # Substrings to remove from layer keys
    substrings_to_remove = ["base_model", "model", "base_layer"]

    # Terms to filter out
    terms_to_filter_out = ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]
    
    for key, value in state_dict.items():
        # Remove specific substrings
        new_key = key
        for substr in substrings_to_remove:
            new_key = new_key.replace(f"{substr}.", "")  # Removing the substring and dot
            new_key = new_key.replace(f"{substr}_", "")  # Removing the substring and underscore
        
        # Skip layers containing specific terms
        if any(term in new_key for term in terms_to_filter_out):
            continue

        new_state_dict[new_key] = value

    return new_state_dict

# Custom function to load compatible parts of the pretrained model
def load_compatible_model(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    model_dict = model.state_dict()
    
    print("preprocessing pretrained state dict...")
    preprocessed_dict = preprocess_state_dict(pretrained_dict)

    # Lists to track issues
    missing_keys = []
    unexpected_keys = []
    size_mismatched_keys = []

    # Filter out mismatched weights
    compatible_dict = {}
    for k, v in preprocessed_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                size_mismatched_keys.append((k, model_dict[k].shape, v.shape))
        else:
            unexpected_keys.append(k)

    # Detect missing keys
    missing_keys = [k for k in model_dict if k not in compatible_dict]

    # Update model dictionary with compatible weights
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)

    # Log details of incompatible keys
    log_incompatible_keys(missing_keys, unexpected_keys, size_mismatched_keys)
    
    return model

def get_lora(model, cfg):

    print("Wrapping model with pretrained LoRA adapter...")
    adapter_config = PeftConfig.from_pretrained(cfg.sample.adapter_config_path)
    adapter_model = PeftModel.from_pretrained(model, cfg.sample.adapter_model_path, config=adapter_config)
    adapter_model.set_adapter("default")
    
    adapter_model.print_trainable_parameters()
    
    print("Freezing all layers...")
    for name, param in model.named_parameters():
        param.requires_grad = False

    return adapter_model

@torch.no_grad()
def layout_to_image_generation(cfg, model_fn, noise_schedule, custom_layout_dict):
    print(custom_layout_dict)

    layout_length = cfg.data.parameters.layout_length

    model_kwargs = {
        'obj_bbox': torch.zeros([1, layout_length, 4]),
        'obj_class': torch.zeros([1, layout_length]).long().fill_(object_name_to_idx['__null__']),
        'is_valid_obj': torch.zeros([1, layout_length])
    }
    model_kwargs['obj_class'][0][0] = object_name_to_idx['__image__']
    model_kwargs['obj_bbox'][0][0] = torch.FloatTensor([0, 0, 1, 1])
    model_kwargs['is_valid_obj'][0][0] = 1.0

    for obj_id in range(1, custom_layout_dict['num_obj']-1):
        obj_bbox = custom_layout_dict['obj_bbox'][obj_id]
        obj_class = custom_layout_dict['obj_class'][obj_id]
        if obj_class == 'pad':
            obj_class = '__null__'

        model_kwargs['obj_bbox'][0][obj_id] = torch.FloatTensor(obj_bbox)
        print(object_name_to_idx)
        model_kwargs['obj_class'][0][obj_id] = object_name_to_idx[obj_class]
        model_kwargs['is_valid_obj'][0][obj_id] = 1

    print(model_kwargs)

    wrappered_model_fn = model_wrapper(
        model_fn,
        noise_schedule,
        is_cond_classifier=False,
        total_N=1000,
        model_kwargs=model_kwargs
    )
    for key in model_kwargs.keys():
        model_kwargs[key] = model_kwargs[key].cuda()

    dpm_solver = DPM_Solver(wrappered_model_fn, noise_schedule)

    x_T = th.randn((1, 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size)).cuda()

    sample = dpm_solver.sample(
        x_T,
        steps=int(cfg.sample.timestep_respacing[0]),
        eps=float(cfg.sample.eps),
        adaptive_step_size=cfg.sample.adaptive_step_size,
        fast_version=cfg.sample.fast_version,
        clip_denoised=False,
        rtol=cfg.sample.rtol
    )  # (B, 3, H, W), B=1

    sample = sample.clamp(-1, 1)

    generate_img = np.array(sample[0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)
    # generate_img = np.transpose(generate_img, (1,0,2))
    print(generate_img.shape)

    print("sampling complete")

    return generate_img

def compute_checksum(file_path, algorithm='sha256'):
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def load_checksum(checksum_file_path):
    with open(checksum_file_path, 'r') as f:
        return f.read().strip()

def verify_checksum(file_path, checksum_file_path, algorithm='sha256'):
    "Verifying checksum..."
    saved_checksum = load_checksum(checksum_file_path)
    computed_checksum = compute_checksum(file_path, algorithm)
    if saved_checksum != computed_checksum:
        raise ValueError(f"Checksum verification failed for {file_path}.\n"
                         f"Saved: {saved_checksum}\n"
                         f"Computed: {computed_checksum}")
    else:
        print(f"Checksum verification successful for {file_path}.")


@torch.no_grad()
def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/COCO-stuff_256x256/LayoutDiffusion-v7_small.yaml')
    parser.add_argument("--share", action='store_true')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    print(OmegaConf.to_yaml(cfg))
    
    is_peft = cfg.sample.is_peft

    print("creating model...")
    model = build_model(cfg)
    model.cuda()

    if cfg.sample.pretrained_model_path:
        print("Loading model from {}".format(cfg.sample.pretrained_model_path))
        checkpoint_path = cfg.sample.pretrained_model_path
        
        # Define the checksum path
        checksum_path = checkpoint_path + ".checksum"
        
        # Verify the checksum before loading the model
        try:
            verify_checksum(checkpoint_path, checksum_path)
        except ValueError as e:
            print(e)
            print("Aborting model load due to checksum mismatch.")
            exit(1)

        # Proceed to load the model
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        try:
            model.load_state_dict(checkpoint, strict=True)
            print('Successfully loaded the entire model')
        except RuntimeError as e:
            print('Not successfully loaded the entire model')
            print('Trying to load part of the model...')
            model = load_compatible_model(model, checkpoint_path)
            
    if is_peft:
        model = get_lora(model, cfg)

    model.cuda()
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
        assert obj_class is not None
        assert obj_bbox is not None

        cond_image, cond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        obj_class = th.ones_like(obj_class).fill_(model.layout_encoder.num_classes_for_layout_object - 1)
        obj_class[:, 0] = 0

        obj_bbox = th.zeros_like(obj_bbox)
        obj_bbox[:, 0] = th.FloatTensor([0, 0, 1, 1])

        is_valid_obj = th.zeros_like(obj_class)
        is_valid_obj[:, 0] = 1.0

        if obj_mask is not None:
            obj_mask = th.zeros_like(obj_mask)
            obj_mask[:, 0] = th.ones(obj_mask.shape[-2:])

        uncond_image, uncond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]
        else:
            return mean

    print("creating diffusion...")

    noise_schedule = NoiseScheduleVP(schedule='linear')

    print('sample method = {}'.format(cfg.sample.sample_method))
    print("sampling...")

    return cfg, model_fn, noise_schedule


if __name__ == "__main__":
    cfg, model_fn, noise_schedule = init()

    demo = get_demo(layout_to_image_generation, cfg, model_fn, noise_schedule)

    demo.launch(share=cfg.share)
