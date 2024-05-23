import cv2
from scipy.io import loadmat
from skimage import img_as_float32, img_as_ubyte
from tqdm import tqdm
noise_path = r'D:\py\MPRNet-main\Denoising\Datasets\test\SIDD\ValidationGtBlocksSrgb.mat'
im_noisy = loadmat(noise_path)['ValidationGtBlocksSrgb']
t = 0
for i in tqdm(range(40)):
    for k in range(32):
        img = im_noisy[i,k,:,:,:]
        cv2.imwrite(f'img/gt/{t}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        t+=1