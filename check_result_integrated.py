import glob
import numpy as np
import torch
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import os
from PIL import Image
from guided_diffusion.custom_util import *
import sys

torch.cuda.set_device(4)


def find_missing_numbers(file_list):
    # Extract the numbers from the file names
    numbers = [int(filename.split('.')[0]) for filename in file_list]

    # Sort the numbers to find the missing ones in order
    numbers.sort()

    # Find the missing numbers
    missing_numbers = [num for num in range(numbers[0], numbers[-1] + 1) if num not in numbers]

    return missing_numbers



def check_all_results(path):
    files = glob.glob(path)
    
    print ("Total files:",len(files))
    psnr_list = []
    ssim_list = []
    fid_list = []
    lpips_list = []
    
    for file_path in files:
        with open(file_path, "r") as file:
            for line in file:
                psnr,ssim,fid,lpips = line.strip().split(",")
                psnr_list.append(float(psnr))
                ssim_list.append(float(ssim))
                fid_list.append(float(fid))
                lpips_list.append(float(lpips))
                
    print ("PSNR= ",round(np.array(psnr_list).mean(),2))
    print ("SSIM= ",round(np.array(ssim_list).mean(),3))    
    print ("FID= ",round(np.array(fid_list).mean(),2))    
    print ("LPIPS= ",round(np.array(lpips_list).mean(),3))     
    
def load_image(image_path, size=None):
    image = Image.open(image_path).convert('RGB')
    if size:
        image = image.resize((size, size), Image.LANCZOS)
    return image

def check_folder(folder1, folder2, limit= 0, image_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    psnrs, ssims, fids, lpipss = [], [], [], []
    files1 = {f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
    files2 = {f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}
    

    common_files = files1.intersection(files2)
    
    common_files = list(common_files)

    common_files.sort()
    
    if limit:
        common_files = common_files[:limit]

    print ("count num:",len(common_files)," in ",folder2)


    if len(common_files) !=1000 and len(common_files) !=100:
        print ("total count is less than 1K")
        print ("real count:", len(common_files))
        print ("missing:",find_missing_numbers(common_files))
        exit(-1)

    
    img1_list =[] 
    img2_list =[]
    for file_name in common_files:
        path1 = os.path.join(folder1, file_name)
        path2 = os.path.join(folder2, file_name)
        #print (path1,path2)

        image1 = load_image(path1, image_size)
        image2 = load_image(path2, image_size)

        img1 = transform(image1).unsqueeze(0).to(device)
        img2 = transform(image2).unsqueeze(0).to(device)

        psnr_value = round(calculate_psnr(normalize(img1), normalize(img2)), 2)
        ssim_value = round(calculate_ssim(normalize(img1), normalize(img2)), 3)
        lpips_value = round(calculate_lpips(normalize(img1), normalize(img2), device), 3)

        #print(psnr_value,ssim_value,fid_value,lpips_value)
        

        psnrs.append(psnr_value)
        ssims.append(ssim_value)
        #fids.append(fid_value)
        lpipss.append(lpips_value)
        img1_list.append(normalize(img1))
        img2_list.append(normalize(img2))
        
    img1_list = torch.cat(img1_list, dim=0)
    img2_list = torch.cat(img2_list, dim=0)
        
    
    fid_value = round(calculate_fid(img1_list, img2_list, device), 2)

    print(f'Average PSNR: {round(np.mean(psnrs),2)}')
    print(f'Average SSIM: {round(np.mean(ssims),3)}')
    print(f'Average FID: {round(np.mean(fid_value),2)}') 
    print(f'Average LPIPS: {round(np.mean(lpipss),3)}')


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py arg1 arg2")
        sys.exit(1)

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    check_folder(arg1,arg2)

if __name__ == "__main__":
    main()