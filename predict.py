import argparse
import logging
import os
from os import listdir
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=(512, 378),
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale=scale_factor,transforms=None))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/CP_epoch200.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', default='/home/ROBARTS/sxing/Needle_segmentation_2D_US/Pytorch-UNet/data/P003_liver_needle_insertion/Needle_Video_02/', metavar='INPUT', nargs='+',
                        help='filenames of input images')

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=(512, 378))

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    # in_files = 'data/test_image/'
    out_files = []
    in_files_l= listdir(in_files)
    if not args.output:
        for f in in_files_l:
            pathsplit = os.path.splitext(f)
            # out_files = 'Out_'+pathsplit[0]+pathsplit[1]
            out_files.append("OUT_{}{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files_l) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    # test_dataset = torchvision.datasets.ImageFolder(root=in_files)
    # testdata_files = torch.utils.data.dataloader(test_dataset)
    in_files_list= listdir(in_files)
    # c = enumerate(a)
    out_file_prefix = '/home/ROBARTS/sxing/Needle_segmentation_2D_US/Pytorch-UNet/data/P003_liver_needle_insertion/Needle_Video_02_Out_epoch200/'
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=1, bilinear=False)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files_list):
        logging.info("\nPredicting image {} ...".format(fn))
        filenames_full = in_files + fn
        img = Image.open(in_files+fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_file_prefix+out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
