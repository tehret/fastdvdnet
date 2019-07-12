#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils import init_logger_test, \
    variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger
import numpy as np

import iio as piio
import os

NUM_IN_FR_EXT = 5  # temporal size of patch
MC_ALGO = 'DeepFlow'  # motion estimation algorithm


def compute_psnr(img1, img2, peak=1):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    x = ((np.array(img1).squeeze() - np.array(img2).squeeze()).flatten())
    return (10 * np.log10(peak**2 / np.mean(x**2)))


def write_file(f, img):
    """
    Write a file f.
    """
    img = np.squeeze(img)
    if f[-4:] == 'tiff' or f[-3:] == 'tif':
        piio.write(f, img)
    else:
        img = np.floor(img + 0.5)
        img[img < 0] = 0
        img[img > 255] = 255
        img = np.asarray(img, dtype=np.uint8)
        piio.write(f, img)


def save_out_seq(seq, save_path, first):
    """Saves the denoised and noisy sequences under save_dir
    """
    seq_len = seq.size()[0]
    for idx in range(seq_len):
        write_file(save_path % (first + idx), seq[idx])


def test_fastdvdnet(**args):
    """Denoises all sequences present in a given folder. Sequences must be stored as numbered
    image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

    Inputs:
            args (dict) fields:
                    "model_file": path to model
                    "test_path": path to sequence to denoise
                    "suffix": suffix to add to output name
                    "max_num_fr_per_seq": max number of frames to load per sequence
                    "noise_sigma": noise level used on test set
                    "dont_save_results: if True, don't save output images
                    "no_gpu": if True, run model on CPU
                    "save_path": where to save outputs as png
                    "gray": if True, perform denoising of grayscale images instead of RGB
    """
    # Start time
    start_time = time.time()

    # If save_path does not exist, create it
    logger = init_logger_test(os.path.dirname(args['save_path']))

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create models
    print('Loading models ...')
    model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    # Load saved weights
    state_temp_dict = torch.load(args['model_file'])
    if args['cuda']:
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    else:
        # CPU mode: remove the DataParallel wrapper
        state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
    model_temp.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()

    with torch.no_grad():
        # process data
        seq = open_sequence(args['test_path'], args[
                            'first'], args['last'], args['already_norm'])
        seq = torch.from_numpy(seq).to(device)
        seq_time = time.time()

        # Add noise
        if not args['already_noisy']:
            noise = torch.empty_like(seq).normal_(
                mean=0, std=args['noise_sigma']).to(device)
            seqn = seq + noise
        else:
            seqn = seq
        noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

        denframes = denoise_seq_fastdvdnet(seq=seqn,
                                           noise_std=noisestd,
                                           temp_psz=NUM_IN_FR_EXT,
                                           model_temporal=model_temp)

    # Compute PSNR and log it
    stop_time = time.time()
    psnr = compute_psnr(denframes.cpu().numpy(), seq.cpu().numpy(), 1.)
    psnr_noisy = compute_psnr(
        seqn.cpu().numpy().squeeze(), seq.cpu().numpy(), 1.)
    loadtime = (seq_time - start_time)
    runtime = (stop_time - seq_time)
    seq_length = seq.size()[0]
    logger.info("Finished denoising {}".format(args['test_path']))
    logger.info("\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s".
                format(seq_length, runtime, loadtime))
    logger.info(
        "\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))

    # Save outputs
    if not args['dont_save_results']:
        # Save sequence
        save_out_seq(denframes.cpu(), args['save_path'], args['first'])

    # close logger
    close_logger(logger)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Denoise a sequence with FastDVDnet")
    parser.add_argument("--model_file", type=str,
                        default="./model.pth",
                        help='path to model of the pretrained denoiser')
    parser.add_argument("--test_path", type=str, default="%03d.png",
                        help='path to sequence to denoise')
    parser.add_argument("--first", type=int, default=1,
                        help='beginning frame index')
    parser.add_argument("--last", type=int, default=100,
                        help='end frame index')
    parser.add_argument("--noise_sigma", type=float,
                        default=25, help='noise level used on test set')
    parser.add_argument(
        "--dont_save_results", action='store_true', help="don't save output images")
    parser.add_argument(
        "--save_noisy", action='store_true', help="save noisy frames")
    parser.add_argument(
        "--no_gpu", action='store_true', help="run model on CPU")
    parser.add_argument(
        "--already_noisy", action='store_true', help="don't add noise to the input frames")
    parser.add_argument(
        "--already_norm", action='store_true', help="don't add normalize to the input frames")
    parser.add_argument(
        "--save_path", type=str, default='deno_%03d.png', help='where to save the output')
    parser.add_argument("--gray", action='store_true',
                        help='perform denoising of grayscale images instead of RGB')

    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))
