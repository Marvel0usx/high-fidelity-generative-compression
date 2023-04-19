import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import glob
import PIL
import time
import re
import matplotlib


matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 


# end dir with forward slash
def seed_sweep(input_dir, output_dir, model_dir):
    models = os.listdir(model_dir)
    for model in models:
        os.makedirs(output_dir + "/" + model)
        input_arg  = " -i " + input_dir
        ckpt_arg   = " -ckpt " + model
        output_arg = " -o " + output_dir + "/" + model + "/"
        command = "python3 /kaggle/working/high-fidelity-generative-compression/compress.py --save " \
                    + input_arg + output_arg + ckpt_arg
        os.system(command)
        # avoid GPU overload
        time.sleep(120)


def plot_comparison_triplet(original_dir, hific_dir, ours_dir, scale_to=None):
    # get image triplet
    original_images = os.listdir(original_dir)

    for original_name in original_images[:2]:
        # strip off extension
        fname  = os.path.basename(original_name)[:-4]
        hific_img_path = glob.glob(hific_dir + "/" + fname + "*.png")[0]
        our_img_path   = glob.glob(ours_dir + "/" + fname + "_RECON_" + "*.png")[0]
        original_img_path = original_dir + "/" + original_name
        
        img = PIL.Image.open(original_img_path)
        W, H = img.size
        filesize = os.path.getsize(original_img_path)
        origin_bpp = filesize * 8. / (H * W)

        bpp_pattern = r"\_(\d\.\d+)bpp\.png"
        hific_bpp = re.findall(bpp_pattern, hific_img_path)[0]
        ours_bpp = re.findall(bpp_pattern, our_img_path)[0]
        
        origi_img = plt.imread(original_img_path)
        hific_img = plt.imread(hific_img_path)
        ours_img  = plt.imread(our_img_path)
        
        fig, ax = plt.subplots(1, 3, figsize=(75, 75))
        
        ax[0].set_xlabel(f"{origin_bpp:.3f}bpp", fontsize = 45)
        ax[0].imshow(origi_img)
        
        ax[1].set_xlabel(hific_bpp + "bpp", fontsize = 45)
        ax[1].imshow(hific_img)
        
        ax[2].set_xlabel(ours_bpp + "bpp", fontsize = 45) 
        ax[2].imshow(ours_img)


def plot_loss(loss_pickle_path):
    with open(loss_pickle_path, "rb") as fin:
        loss = pickle.load(fin)
        print(loss)
