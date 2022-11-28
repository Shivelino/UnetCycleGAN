#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from nets.cyclegan import Generator, Discriminator
from nets.common_modules import ContextUnet
from utils.dataloader import CycleGanDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str,
                        # default='datasets/flower41toflower49/',
                        default='datasets/flower51toflower98/',
                        help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str,
                        # default=r'E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\tmp_store\output_41_49_unet\netG_A2B_unet_190.pth',
                        default=r'E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\tmp_store\output_51_98_unet\netG_A2B_unet_190.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str,
                        # default=r'E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\tmp_store\output_41_49_unet\netG_B2A_unet_190.pth',
                        default=r'E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\tmp_store\output_51_98_unet\netG_B2A_unet_190.pth',
                        help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = ContextUnet(opt.input_nc)
    netG_B2A = ContextUnet(opt.output_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B), False)
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A), False)

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(CycleGanDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    if not os.path.exists('output/pil/A'):
        os.makedirs('output/pil/A')
    if not os.path.exists('output/pil/B'):
        os.makedirs('output/pil/B')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)
        # fake_B = 0.33 * (netG_A2B(real_A).data + 2.0)
        # fake_A = 0.33 * (netG_B2A(real_B).data + 2.0)
        # fake_B = netG_A2B(real_A).data
        # fake_A = netG_B2A(real_B).data

        # Save image files
        save_image(fake_A, 'output/A/%04d.png' % (i + 1))
        save_image(fake_B, 'output/B/%04d.png' % (i + 1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

    sys.stdout.write('\n')
    ###################################
