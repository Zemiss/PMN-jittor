import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
import os
import glob
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import gc
from PIL import Image
import cv2
import time
import socket
import scipy
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from functools import wraps
from tqdm import tqdm
import exifread
import rawpy
import math
import yaml
import pickle
import warnings
import h5py
import pickle as pkl

fn_time = {}

def timestamp(time_points, n):
    time_points[n] = time.time()
    return time_points[n] - time_points[n-1]

def fn_timer(function, print_log=False):
    @wraps(function)
    def function_timer(*args, **kwargs):
        global fn_timer
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        if print_log:
            print ("Total time running %s: %.6f seconds" %
                (function.__name__, t1-t0))
        if function.__name__ in fn_time :
            fn_time[function.__name__] += t1-t0
        else:
            fn_time[function.__name__] = t1-t0
        return result
    return function_timer

def log(string, log=None, str_mode=False, end='\n', notime=False):
    if jt.rank > 0: return
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    print(log_string)
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')
    if str_mode:
        return string+end

def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns

def metrics_recorder(file, names, psnrs, ssims):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            metrics = pkl.load(f)
    else:
        metrics = {}
    for name, psnr, ssim in zip(names, psnrs, ssims):
        metrics[name] = [psnr, ssim]
    with open(file, 'wb') as f:
        pkl.dump(metrics, f)
    return metrics
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', log=True, last_epoch=0):
        self.name = name
        self.fmt = fmt
        self.log = log
        self.history = []
        self.last_epoch = last_epoch
        self.history_init_flag = False
        self.reset()

    def reset(self):
        if self.log:
            try:
                if self.avg>0: self.history.append(self.avg)
            except:
                pass
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def sync(self):
        if jt.in_mpi:
            self.sum = jt.array([self.sum]).mpi_all_reduce('sum').item()
            self.count = jt.array([self.count]).mpi_all_reduce('sum').item()
            self.avg = self.sum / self.count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def plot_history(self, savefile='log.jpg', logfile='log.pkl'):
        # 读取老log
        if os.path.exists(logfile) and not self.history_init_flag:
            self.history_init_flag = True
            with open(logfile, 'rb') as f:
                history_old = pickle.load(f)
                if self.last_epoch: # 为0则重置
                    self.history = history_old + self.history[:self.last_epoch]
        # 记录log
        with open(logfile, 'wb') as f:
            pickle.dump(self.history, f)
        # 画图
        plt.figure(figsize=(12,9))
        plt.title(f'{self.name} log')
        x = list(range(len(self.history)))
        plt.plot(x, self.history)
        plt.xlabel('Epoch')
        plt.ylabel(self.name)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def pkl_convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }

def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.color_matrix[:3, :3].astype(np.float32)
    if ccm[0,0] == 0:
        ccm = np.eye(3, dtype=np.float32)
    return wb, ccm

def get_ISO_ExposureTime(filepath):
    # 不限于RAW，RGB图片也适用
    raw_file = open(filepath, 'rb')
    exif_file = exifread.process_file(raw_file, details=False, strict=True)
    # 获取曝光时间
    if 'EXIF ExposureTime' in exif_file:
        exposure_str = exif_file['EXIF ExposureTime'].printable
    else:
        exposure_str = exif_file['Image ExposureTime'].printable
    if '/' in exposure_str:
        fenmu = float(exposure_str.split('/')[0])
        fenzi = float(exposure_str.split('/')[-1])
        exposure = fenmu / fenzi
    else:
        exposure = float(exposure_str)
    # 获取ISO
    if 'EXIF ISOSpeedRatings' in exif_file:
        ISO_str = exif_file['EXIF ISOSpeedRatings'].printable
    else:
        ISO_str = exif_file['Image ISOSpeedRatings'].printable
    if '/' in ISO_str:
        fenmu = float(ISO_str.split('/')[0])
        fenzi = float(ISO_str.split('/')[-1])
        ISO = fenmu / fenzi
    else:
        ISO = float(ISO_str)
    info = {'ISO':int(ISO), 'ExposureTime':exposure, 'name':filepath.split('/')[-1]}
    return info

def load_weights(model, pretrained_dict, by_name=False):
    """Load pretrained weights to Jittor model"""
    model_dict = model.state_dict()
    
    # 1. filter out unnecessary keys
    tsm_replace = []
    for k in pretrained_dict:
        if 'tsm_shift' in k:
            k_new = k.replace('tsm_shift', 'tsm_buffer')
            tsm_replace.append((k, k_new))
    for k, k_new in tsm_replace:
        pretrained_dict[k_new] = pretrained_dict[k]
    
    if by_name:
        del_list = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape != pretrained_dict[k].shape:
                    del_list.append(k)
                    log(f'Warning:  "{k}":{pretrained_dict[k].shape}->{model_dict[k].shape}')
                # Convert numpy to jittor if needed
                if isinstance(v, np.ndarray):
                    pretrained_dict[k] = jt.array(v)
            else:
                del_list.append(k)
                log(f'Warning:  "{k}" is not exist and has been deleted!!')
        for k in del_list:
            del pretrained_dict[k]
    
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def tensor_dim5to4(tensor):
    batchsize, crops, c, h, w = tensor.shape
    tensor = tensor.reshape(batchsize*crops, c, h, w)
    return tensor

def tensor_dim6to5(tensor):
    batchsize, crops, t, c, h, w = tensor.shape
    tensor = tensor.reshape(batchsize*crops, t, c, h, w)
    return tensor

def frame_index_splitor(nframes=1, pad=True, reflect=True):
    # [b, 7, c, h ,w]
    r = nframes // 2
    length = 7 if pad else 8-nframes
    frames = []
    for i in range(length):
        frames.append([None]*nframes)
    if pad:
        for i in range(7):
            for k in range(nframes):
                frames[i][k] = i+k-r
    else:
        for i in range(8-nframes):
            for k in range(nframes):
                frames[i][k] = i+k
    if reflect:
        frames = num_reflect(frames,0,6)
    else:
        frames = num_clip(frames, 0, 6)
    return frames

def multi_frame_loader(frames, index, gt=False, keepdims=False):
    loader = []
    for ind in index:
        imgs = []
        if gt:
            r = len(index[0]) // 2
            tensor = frames[:,ind[r],:,:,:]
            if keepdims:
                tensor = tensor.unsqueeze(dim=1)
        else:
            for i in ind:
                imgs.append(frames[:,i,:,:,:])
            tensor = jt.stack(imgs, dim=1)
        loader.append(tensor)
    return jt.stack(loader, dim=0)

def num_clip(nums, mininum, maxinum):
    nums = np.array(nums)
    nums = np.clip(nums, mininum, maxinum)
    return nums

def num_reflect(nums, mininum, maxinum):
    nums = np.array(nums)
    nums = np.abs(nums-mininum)
    nums = maxinum-np.abs(maxinum-nums)
    return nums

def get_host_with_dir(dataset_name=''):
    hostname = socket.gethostname()
    log(f"User's hostname is '{hostname}'")
    if hostname == 'fenghansen':
        host = '/data'
    elif hostname == 'DESKTOP-FCAMIOQ':
        host = 'F:/datasets'
    elif hostname == 'BJ-DZ0101767':
        host = 'F:/Temp'
    else:
        host = '/data'
    return hostname, host + dataset_name, jt.flags.use_cuda

def scale_down(img):
    return np.float32(img) / 255.

def scale_up(img):
    return np.uint8(img * 255.)

def feature_vis(tensor, name='out', save=False):
    """Visualize features from Jittor tensor"""
    if isinstance(tensor, jt.Var):
        feature = tensor.detach().numpy().transpose(0,2,3,1)
    else:
        feature = tensor.transpose(0,2,3,1)
    
    if save:
        for i in range(len(feature)):
            cv2.imwrite(f'./test/{name}_{i}.png', np.uint8(feature[i,:,:,::-1]*255))
    return feature

def bayer2rggb(bayer):
    H, W = bayer.shape
    return bayer.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)

def rggb2bayer(rggb):
    H, W, _ = rggb.shape
    return rggb.reshape(H, W, 2, 2).transpose(0, 2, 1, 3).reshape(H*2, W*2)

def repair_bad_pixels(raw, bad_points, method='median'):
    fixed_raw = bayer2rggb(raw)
    for i in range(4):
        fixed_raw[:,:,i] = cv2.medianBlur(fixed_raw[:,:,i],3)
    fixed_raw = rggb2bayer(fixed_raw)
    for p in bad_points:
        raw[p[0],p[1]] = fixed_raw[p[0],p[1]]
    return raw

def img4c_to_RGB(img4c, metadata=None, gamma=2.2):
    h,w,c = img4c.shape
    H = h * 2
    W = w * 2
    raw = np.zeros((H,W), np.float32)
    red_gain = metadata['red_gain'] if metadata is not None else 1
    blue_gain = metadata['blue_gain'] if metadata is not None else 1
    rgb_gain = metadata['rgb_gain'] if metadata is not None else 1
    raw[0:H:2,0:W:2] = img4c[:,:,0] * red_gain # R
    raw[0:H:2,1:W:2] = img4c[:,:,1] # G1
    raw[1:H:2,1:W:2] = img4c[:,:,2] * blue_gain # B
    raw[1:H:2,0:W:2] = img4c[:,:,3] # G2
    raw = np.clip(raw * rgb_gain, 0, 1)
    white_point = 16383
    raw = raw * white_point
    img = cv2.cvtColor(raw.astype(np.uint16), cv2.COLOR_BAYER_BG2RGB_EA) / white_point
    ccms = np.array([[ 1.7479, -0.7025, -0.0455],
        [-0.2163,  1.5111, -0.2948],
        [ 0.0054, -0.6514,  1.6460]])
    img = img[:, :, None, :]
    ccms = ccms[None, None, :, :]
    img = np.sum(img * ccms, axis=-1)
    img = np.clip(img, 0, 1) ** (1/gamma)
    return img

def FastGuidedFilter(p,I,d=7,eps=1):
    p_lr = cv2.resize(p, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    I_lr = cv2.resize(I, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    mu_p = cv2.boxFilter(p_lr, -1, (d, d)) 
    mu_I = cv2.boxFilter(I_lr,-1, (d, d)) 
    
    II = cv2.boxFilter(np.multiply(I_lr,I_lr), -1, (d, d)) 
    Ip = cv2.boxFilter(np.multiply(I_lr,p_lr), -1, (d, d))
    
    var = II-np.multiply(mu_I,mu_I)
    cov = Ip-np.multiply(mu_I,mu_p)
    
    a = cov / (var + eps)
    
    b = mu_p - np.multiply(a,mu_I)
    mu_a = cv2.resize(cv2.boxFilter(a, -1, (d, d)), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
    mu_b = cv2.resize(cv2.boxFilter(b, -1, (d, d)), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
    
    dstImg = np.multiply(mu_a, I) + mu_b
    
    return dstImg

def GuidedFilter(p,I,d=7,eps=1):
    mu_p = cv2.boxFilter(p, -1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    mu_I = cv2.boxFilter(I,-1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    
    II = cv2.boxFilter(np.multiply(I,I), -1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    Ip = cv2.boxFilter(np.multiply(I,p), -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    
    var = II-np.multiply(mu_I,mu_I)
    cov = Ip-np.multiply(mu_I,mu_p)
    
    a = cov / (var + eps)
    
    b = mu_p - np.multiply(a,mu_I)
    mu_a = cv2.boxFilter(a, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    mu_b = cv2.boxFilter(b, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    
    dstImg = np.multiply(mu_a, I) + mu_b
    
    return dstImg

def plot_sample(img_lr, img_dn, img_hr, filename='result', model_name='Unet', 
                epoch=-1, print_metrics=False, save_plot=True, save_path='./', res=None):
    if np.max(img_hr) <= 1:
        # 变回uint8
        img_lr = scale_up(img_lr)
        img_dn = scale_up(img_dn)
        img_hr = scale_up(img_hr)
    # 计算PSNR和SSIM
    if res is None:
        psnr = []
        ssim = []
        psnr.append(compare_psnr(img_hr, img_lr))
        psnr.append(compare_psnr(img_hr, img_dn))
        ssim.append(compare_ssim(img_hr, img_lr, channel_axis=-1))
        ssim.append(compare_ssim(img_hr, img_dn, channel_axis=-1))
        psnr.append(-1)
        ssim.append(-1)
    else:
        psnr = [res[0], res[2], -1]
        ssim = [res[1], res[3], -1]
    # Images and titles
    images = {
        'Noisy Image': img_lr,
        model_name: img_dn,
        'Ground Truth': img_hr
    }
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img)
        axes[i].set_title("{} - {} - psnr:{:.2f} - ssim{:.4f}".format(title, img.shape, psnr[i], ssim[i]))
        axes[i].axis('off')
    plt.suptitle('{} - Epoch: {}'.format(filename, epoch))
    if print_metrics:
        log('PSNR:', psnr)
        log('SSIM:', ssim)
    # Save directory
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    savefile = os.path.join(save_path, "{}-Epoch{}.jpg".format(filename, epoch))
    if save_plot:
        denoisedfile = os.path.join(save_path, "{}_denoised.png".format(filename))
        cv2.imwrite(denoisedfile, img_dn[:,:,::-1])
        fig.savefig(savefile, bbox_inches='tight')
        plt.close()
    return psnr, ssim

def save_picture(img_sr, save_path='./images/test', frame_id='0000'):
    # 变回uint8
    img_sr = scale_up(img_sr.transpose(1,2,0))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.imsave(os.path.join(save_path, frame_id+'.png'), img_sr)
    gc.collect()

def test_output_rename(root_dir):
    for dirs in os.listdir(root_dir):
        dirpath = root_dir + '/' + dirs
        f = os.listdir(dirpath)
        end = len(f)
        for i in range(len(f)):
            frame_id = int(f[end-i-1][:4])
            old_file = os.path.join(dirpath, "%04d.png" % frame_id)
            new_file = os.path.join(dirpath, "%04d.png" % (frame_id + 1))
            os.rename(old_file, new_file)
        log(f"path |{dirpath}|'s rename has finished...")

def datalist_rename(root_dir):
    src_file = os.path.join(root_dir, 'sep_testlist.txt')
    dst_file = os.path.join(root_dir, 'sep_evallist.txt')
    sub_dirs = []
    fw = open(dst_file, 'w')
    with open(src_file, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        for sub_path in lines:
            if sub_path[:5] in sub_dirs: continue
            sub_dirs.append(sub_path[:5])
            print(sub_path, file=fw)
    fw.close()
    return sub_dirs

def tensor2im(image_tensor, visualize=False, video=False):
    """Convert Jittor tensor to numpy image"""
    if isinstance(image_tensor, jt.Var):
        image_tensor = image_tensor.detach()
    
    if visualize:                
        image_tensor = image_tensor[:, 0:3, ...]

    if not video: 
        image_numpy = image_tensor[0].numpy() if isinstance(image_tensor, jt.Var) else image_tensor[0]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_tensor.numpy() if isinstance(image_tensor, jt.Var) else image_tensor
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy

def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:
        psnr = compare_psnr(Y, X, data_range=data_range)
        try:
            ssim = compare_ssim(Y, X, data_range=data_range, multichannel=True)
        except (ValueError, TypeError):
            ssim = 0
        return {'PSNR':psnr, 'SSIM': ssim}
    else:
        raise NotImplementedError

def bayer2rows(bayer):
    H, W = bayer.shape
    return np.stack((bayer[0:H:2], bayer[1:H:2]))

def rows2bayer(rows):
    c, H, W = rows.shape
    bayer = np.empty((H*2, W))
    bayer[0:H*2:2] = rows[0]
    bayer[1:H*2:2] = rows[1]
    return bayer

def dataload(path):
    suffix = path[-4:].lower()
    if suffix in ['.arw','.dng']:
        try:
            data = rawpy.imread(path).raw_image_visible
        except rawpy._rawpy.LibRawTooBigError:
            print(f"Image {path} is too big, attempting to resize...")
            img = Image.open(path)
            img.thumbnail((img.width // 2, img.height // 2))
            img.save("temp_resized.png")
            data = rawpy.imread("temp_resized.png").raw_image_visible
    elif suffix in ['.npy']:
        data = np.load(path)
    elif suffix in ['.jpg', '.png', '.bmp', 'tiff']:
        data = cv2.imread(path)
    return data

def row_denoise(path, iso, data=None):
    if data is None:
        raw = dataload(path)
    else:
        raw = data
    raw = bayer2rows(raw)
    raw_denoised = raw.copy()
    for i in range(len(raw)):
        rows = raw[i].mean(axis=1)
        rows2 = rows.reshape(1, -1)
        rows2 = cv2.bilateralFilter(rows2, 25, sigmaColor=10, sigmaSpace=1+iso/200, borderType=cv2.BORDER_REPLICATE)[0]
        row_diff = rows-rows2
        raw_denoised[i] = raw[i] - row_diff.reshape(-1, 1)
    raw = rows2bayer(raw)
    raw_denoised = rows2bayer(raw_denoised)
    return raw_denoised

def pth_transfer(src_path='/data/ELD/checkpoints/sid-ours-inc4/model_200_00257600.pt',
                dst_path='checkpoints/SonyA7S2_Official.pth',
                reverse=False):
    """Transfer model weights (supports both PyTorch and Jittor)"""
    try:
        # Try loading with Jittor first
        model_src = jt.load(src_path)
        if reverse:
            model_dst = jt.load(dst_path)
            model_src['netG'] = model_dst
            save_dir = os.path.join('pth_transfer', os.path.basename(dst_path)[9:-15])
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(src_path))
            jt.save(model_src, save_path)
        else:
            model_src = model_src['netG']
            jt.save(model_src, dst_path)
    except:
        # Fallback to PyTorch if available
        import torch
        model_src = torch.load(src_path, map_location='cpu')
        if reverse:
            model_dst = torch.load(dst_path, map_location='cpu')
            model_src['netG'] = model_dst
            save_dir = os.path.join('pth_transfer', os.path.basename(dst_path)[9:-15])
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(src_path))
            torch.save(model_src, save_path)
        else:
            model_src = model_src['netG']
            torch.save(model_src, dst_path)


if __name__ == '__main__':
    print("Testing Jittor Utils Functions...")
    print("=" * 80)
    
    # Test 1: AverageMeter
    print("\n1. Testing AverageMeter:")
    meter = AverageMeter('PSNR', fmt=':.4f')
    for i in range(10):
        meter.update(30 + np.random.rand())
    print(f"   {meter}")
    print(f"   Average: {meter.avg:.4f}")
    
    # Test 2: Jittor tensor operations
    print("\n2. Testing Jittor tensor operations:")
    jt.flags.use_cuda = 1
    print(f"   CUDA available: {jt.flags.use_cuda}")
    
    # Create test tensors
    x = jt.randn(2, 3, 64, 64)
    print(f"   Created tensor shape: {x.shape}")
    
    # Test tensor2im
    img_np = tensor2im(x)
    print(f"   tensor2im output shape: {img_np.shape}")
    
    # Test 3: Feature visualization
    print("\n3. Testing feature_vis:")
    feature = jt.randn(1, 16, 32, 32)
    vis_output = feature_vis(feature, name='test', save=False)
    print(f"   Feature vis output shape: {vis_output.shape}")
    
    # Test 4: Image scale operations
    print("\n4. Testing scale_up and scale_down:")
    test_img = np.random.rand(64, 64, 3).astype(np.float32)
    scaled_down = scale_down(test_img * 255)
    scaled_up = scale_up(scaled_down)
    print(f"   Original range: [0, 255], Scaled down: [{scaled_down.min():.2f}, {scaled_down.max():.2f}]")
    print(f"   Scaled up range: [{scaled_up.min()}, {scaled_up.max()}]")
    
    # Test 5: Bayer pattern operations
    print("\n5. Testing bayer2rggb and rggb2bayer:")
    bayer = np.random.randint(0, 4096, (100, 100), dtype=np.uint16)
    rggb = bayer2rggb(bayer)
    bayer_reconstructed = rggb2bayer(rggb)
    print(f"   Bayer shape: {bayer.shape}")
    print(f"   RGGB shape: {rggb.shape}")
    print(f"   Reconstructed Bayer shape: {bayer_reconstructed.shape}")
    print(f"   Reconstruction accurate: {np.allclose(bayer, bayer_reconstructed)}")
    
    # Test 6: Quality assessment
    print("\n6. Testing quality_assess:")
    img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img2 = img1 + np.random.randint(-10, 10, (64, 64, 3), dtype=np.int16)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    metrics = quality_assess(img2, img1)
    print(f"   PSNR: {metrics['PSNR']:.2f} dB")
    print(f"   SSIM: {metrics['SSIM']:.4f}")
    
    # Test 7: Guided Filter
    print("\n7. Testing GuidedFilter:")
    p = np.random.rand(64, 64).astype(np.float32)
    I = np.random.rand(64, 64).astype(np.float32)
    filtered = GuidedFilter(p, I, d=7, eps=0.01)
    print(f"   Input shape: {p.shape}")
    print(f"   Filtered output shape: {filtered.shape}")
    
    # Test 8: Frame index splitor
    print("\n8. Testing frame_index_splitor:")
    indices = frame_index_splitor(nframes=3, pad=True, reflect=True)
    print(f"   Frame indices (nframes=3):")
    for i, idx in enumerate(indices):
        print(f"      Frame {i}: {idx}")
    
    # Test 9: Load weights (mock test)
    print("\n9. Testing load_weights:")
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        
        def execute(self, x):
            return self.conv(x)
    
    model = SimpleModel()
    # Create mock pretrained dict
    pretrained_dict = {
        'conv.weight': jt.randn(16, 3, 3, 3),
        'conv.bias': jt.randn(16)
    }
    model = load_weights(model, pretrained_dict, by_name=True)
    print(f"   Model loaded successfully")
    print(f"   Model parameters: {len(list(model.parameters()))}")
    
    # Test 10: Hostname and directory
    print("\n10. Testing get_host_with_dir:")
    hostname, host_dir, use_cuda = get_host_with_dir('/SID')
    print(f"   Hostname: {hostname}")
    print(f"   Host directory: {host_dir}")
    print(f"   CUDA enabled: {use_cuda}")
    
    # Test 11: Metrics recorder
    print("\n11. Testing metrics_recorder:")
    test_file = 'test_metrics.pkl'
    names = ['img1', 'img2', 'img3']
    psnrs = [30.5, 32.1, 28.9]
    ssims = [0.92, 0.94, 0.89]
    metrics = metrics_recorder(test_file, names, psnrs, ssims)
    print(f"   Metrics recorded for {len(metrics)} images")
    print(f"   Sample: img1 -> PSNR={metrics['img1'][0]:.2f}, SSIM={metrics['img1'][1]:.4f}")
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Test 12: Tensor dimension operations
    print("\n12. Testing tensor dimension operations:")
    tensor_5d = jt.randn(2, 3, 4, 32, 32)
    tensor_4d = tensor_dim5to4(tensor_5d)
    print(f"   5D tensor shape: {tensor_5d.shape}")
    print(f"   Converted to 4D: {tensor_4d.shape}")
    
    tensor_6d = jt.randn(2, 3, 7, 4, 32, 32)
    tensor_5d = tensor_dim6to5(tensor_6d)
    print(f"   6D tensor shape: {tensor_6d.shape}")
    print(f"   Converted to 5D: {tensor_5d.shape}")
    
    # Test 13: Number operations
    print("\n13. Testing num_clip and num_reflect:")
    nums = [[-5, 0, 3, 5, 8, 12]]
    clipped = num_clip(nums, 0, 10)
    reflected = num_reflect(nums, 0, 10)
    print(f"   Original: {nums[0]}")
    print(f"   Clipped [0, 10]: {clipped[0].tolist()}")
    print(f"   Reflected [0, 10]: {reflected[0].tolist()}")
    
    # Test 14: Plot sample (without saving)
    print("\n14. Testing plot_sample:")
    img_lr = np.random.rand(64, 64, 3).astype(np.float32)
    img_dn = np.random.rand(64, 64, 3).astype(np.float32)
    img_hr = np.random.rand(64, 64, 3).astype(np.float32)
    
    # Create temporary directory for test
    test_dir = './test_output'
    os.makedirs(test_dir, exist_ok=True)
    
    psnr, ssim = plot_sample(img_lr, img_dn, img_hr, 
                            filename='test_img',
                            model_name='TestModel',
                            epoch=1,
                            save_plot=True,
                            save_path=test_dir)
    print(f"   PSNR values: {psnr}")
    print(f"   SSIM values: {ssim}")
    print(f"   Plot saved to: {test_dir}")
    
    # Clean up test directory
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("\n" + "=" * 80)
    print("All utils tests completed successfully!")
    print("=" * 80)
    
    # Additional test: Timer decorator
    print("\n15. Testing fn_timer decorator:")
    
    @fn_timer
    def slow_function():
        import time
        time.sleep(0.1)
        return "Done"
    
    result = slow_function()
    print(f"   Function result: {result}")
    print(f"   Function timing recorded in fn_time: {fn_time.get('slow_function', 0):.4f}s")