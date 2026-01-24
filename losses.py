import jittor as jt
from jittor import nn
import numpy as np

# Jittor will automatically use GPU if available
Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = jt.array(Sobel).float32()
Robert = jt.array(Robert).float32()

def gamma(x, clip=True, gamma_val=2.2):
    if clip: # prevent numerical instability
        x = jt.maximum(x, 1e-6)
    return jt.pow(x, 1/gamma_val)

def norm(gradient_orig):
    grad_min = gradient_orig.min()
    grad_max = gradient_orig.max()
    grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 0.0001)
    return grad_norm


# 已测试本模块没有问题，作用为提取一阶导数算子滤波图（边缘图）
def gradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = nn.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = nn.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = jt.abs(nn.conv2d(maps, kernel, padding=0))
    return gradient_orig


def Pyramid_Sample(img, max_scale=8):
    imgs = []
    sample = img
    power = 1
    while 2**power <= max_scale:
        sample = nn.avg_pool2d(sample, 2, 2)
        imgs.append(sample)
        power += 1
    return imgs


def Pyramid_Loss(lows, highs, loss_fn=None, rate=1., norm=True):
    if loss_fn is None:
        loss_fn = lambda x, y: jt.abs(x - y).mean()
    
    losses = []
    for low, high in zip(lows, highs):
        losses.append( loss_fn(low, high) )
    pyramid_loss = 0
    scale = 0
    lam = 1
    for i, loss in enumerate(losses):
        pyramid_loss += loss * lam
        scale += lam
        lam = lam * rate
    if norm:
        pyramid_loss = pyramid_loss / scale
    return pyramid_loss

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def execute(self, low, high):
        diff = low - high
        error = jt.sqrt(diff * diff + self.eps)
        loss = error.mean()
        return loss

class Unet_Loss(nn.Module):
    def __init__(self, charbonnier=False):
        super().__init__()
        if charbonnier:
            self.l1_loss = L1_Charbonnier_loss()
        else:
            self.l1_loss = lambda x, y: jt.abs(x - y).mean()

    def grad_loss(self, low, high):
        grad_x = jt.abs(gradient(low, 'x') - gradient(high, 'x'))
        grad_y = jt.abs(gradient(low, 'y') - gradient(high, 'y'))
        grad_norm = (grad_x + grad_y).mean()
        return grad_norm
    
    def pyramid_loss(self, low, high):
        h2, h4, h8 = Pyramid_Sample(high, max_scale=8)
        l2, l4, l8 = Pyramid_Sample(low, max_scale=8)
        loss = Pyramid_Loss([low, l2, l4, l8], [high, h2, h4, h8], loss_fn=self.loss, rate=0.5, norm=True)
        return loss

    def loss(self, low, high):
        # loss_grad = self.grad_loss(low, high)
        if callable(self.l1_loss):
            loss_recon = self.l1_loss(low, high)
        else:
            loss_recon = self.l1_loss.execute(low, high)
        # loss_recon += self.l1_loss(gamma(low), gamma(high))
        # loss_recon /= 2
        return loss_recon# + loss_grad

    def execute(self, low, high, pyramid=False):
        if pyramid:
            loss = self.pyramid_loss(low, high)
        else:
            loss = self.loss(low, high)
        # loss_recon = jt.abs(low - high).mean()
        # loss_grad = self.grad_loss(low, high)
        # loss_enhance = self.mutual_consistency(low, high, hook)
        return loss

class Unet_dpsv_Loss(Unet_Loss):
    def __init__(self, charbonnier=False):
        super().__init__(charbonnier)

    def execute(self, output, target):
        scale = 2 ** (len(output) - 1)
        target = [target,] + Pyramid_Sample(target, max_scale=scale)
        # loss_restore = self.loss(output, target)
        loss_restore = Pyramid_Loss(output, target,
                                    loss_fn=self.loss, rate=1, norm=False)
        return loss_restore

class Unet_dpsv_Loss_up(Unet_Loss):
    def __init__(self, charbonnier=False):
        super().__init__(charbonnier)

    def execute(self, output, target):
        scale = 2 ** (len(output) - 2)
        target = [target, target,] + Pyramid_Sample(target, max_scale=scale)
        # loss_restore = self.loss(output, target)
        loss_restore = Pyramid_Loss(output, target,
                                    loss_fn=self.loss, rate=1, norm=False)
        return loss_restore

def PSNR_Loss(low, high):
    shape = low.shape
    if len(shape) <= 3:
        psnr = -10.0 * jt.log(jt.pow(high-low, 2).mean()) / jt.log(jt.array(10.0))
    else:
        psnr = jt.zeros(shape[0])
        for i in range(shape[0]):
            psnr[i] = -10.0 * jt.log(jt.pow(high[i]-low[i], 2).mean()) / jt.log(jt.array(10.0))
        # print(psnr)
        psnr = psnr.mean()# / shape[0]
    return psnr 

class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def execute(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = jt.array(self.w).float32()

    def transform(self, img):
        patches = nn.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / jt.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = (dist / (0.1 + dist)).mean(1, keepdims=True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = jt.ones(n, 1, h - 2 * padding, w - 2 * padding).float32()
        mask = nn.pad(inner, [padding] * 4)
        return mask

    def execute(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = jt.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float32()
        self.kernelY = self.kernelX.clone().transpose(1, 0)
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0)

    def execute(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = jt.concat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = nn.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = nn.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = jt.abs(pred_X-gt_X), jt.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss

class GAN_Loss(nn.Module):
    def __init__(self, mode='RaSGAN'):
        super().__init__()
        self.gan_mode = mode
    
    def bce_with_logits(self, logits, targets):
        """Binary cross entropy with logits implementation for Jittor"""
        # sigmoid(x) = 1 / (1 + exp(-x))
        # BCE_with_logits(x, y) = -y * log(sigmoid(x)) - (1-y) * log(1 - sigmoid(x))
        # Stable version: max(x, 0) - x * y + log(1 + exp(-abs(x)))
        max_val = jt.maximum(logits, 0.0)
        loss = max_val - logits * targets + jt.log(1.0 + jt.exp(-jt.abs(logits)))
        return loss
    
    def execute(self, D_real, D_fake, D_fake_for_G):
        y_ones = jt.ones_like(D_real)
        y_zeros = jt.zeros_like(D_fake)

        if self.gan_mode == 'RSGAN':
            ### Relativistic Standard GAN
            # Discriminator loss
            logits_real = D_real - D_fake
            errD = self.bce_with_logits(logits_real, y_ones)
            loss_D = errD.mean()
            # Generator loss
            logits_fake = D_fake_for_G - D_real
            errG = self.bce_with_logits(logits_fake, y_ones)
            loss_G = errG.mean()
        elif self.gan_mode == 'SGAN':
            # Real data Discriminator loss
            errD_real = self.bce_with_logits(D_real, y_ones)
            # Fake data Discriminator loss
            errD_fake = self.bce_with_logits(D_fake, y_zeros)
            loss_D = (errD_real + errD_fake).mean() / 2
            # Generator loss
            errG = self.bce_with_logits(D_fake_for_G, y_ones)
            loss_G = errG.mean()
        elif self.gan_mode == 'RaSGAN':
            # Discriminator loss
            errD = (self.bce_with_logits(D_real - D_fake.mean(), y_ones) + 
                    self.bce_with_logits(D_fake - D_real.mean(), y_zeros))/2
            loss_D = errD.mean()
            # Generator loss
            errG = (self.bce_with_logits(D_real - D_fake_for_G.mean(), y_zeros) + 
                    self.bce_with_logits(D_fake_for_G - D_real.mean(), y_ones))/2
            loss_G = errG.mean()
        elif self.gan_mode == 'RaLSGAN':
            # Discriminator loss
            errD = (((D_real - D_fake.mean() - y_ones) ** 2).mean() + 
                    ((D_fake - D_real.mean() + y_ones) ** 2).mean())/2
            loss_D = errD
            # Generator loss (You may want to resample again from real and fake data)
            errG = (((D_real - D_fake_for_G.mean() + y_ones) ** 2).mean() + 
                    ((D_fake_for_G - D_real.mean() - y_ones) ** 2).mean())/2
            loss_G = errG
        
        return loss_D, loss_G


if __name__ == '__main__':
    print("Testing Jittor Loss Functions...")
    print("=" * 60)
    
    # Set random seed for reproducibility
    jt.set_global_seed(42)
    
    # Create dummy input tensors
    batch_size = 2
    channels = 3
    height, width = 64, 64
    
    pred = jt.randn(batch_size, channels, height, width)
    target = jt.randn(batch_size, channels, height, width)
    
    print(f"\nInput shapes: pred={pred.shape}, target={target.shape}")
    print("-" * 60)
    
    # Test L1_Charbonnier_loss
    print("\n1. Testing L1_Charbonnier_loss:")
    l1_char_loss = L1_Charbonnier_loss()
    loss_val = l1_char_loss.execute(pred, target)
    print(f"   L1 Charbonnier Loss: {loss_val.item():.6f}")
    
    # Test Unet_Loss
    print("\n2. Testing Unet_Loss:")
    unet_loss = Unet_Loss(charbonnier=False)
    loss_val = unet_loss.execute(pred, target, pyramid=False)
    print(f"   Unet Loss (no pyramid): {loss_val.item():.6f}")
    
    loss_val = unet_loss.execute(pred, target, pyramid=True)
    print(f"   Unet Loss (with pyramid): {loss_val.item():.6f}")
    
    # Test Unet_Loss with Charbonnier
    print("\n3. Testing Unet_Loss with Charbonnier:")
    unet_loss_char = Unet_Loss(charbonnier=True)
    loss_val = unet_loss_char.execute(pred, target)
    print(f"   Unet Loss (Charbonnier): {loss_val.item():.6f}")
    
    # Test Unet_dpsv_Loss
    print("\n4. Testing Unet_dpsv_Loss:")
    dpsv_loss = Unet_dpsv_Loss(charbonnier=False)
    # Create multi-scale outputs (4 scales)
    outputs = [pred, 
               nn.avg_pool2d(pred, 2, 2),
               nn.avg_pool2d(pred, 4, 4),
               nn.avg_pool2d(pred, 8, 8)]
    loss_val = dpsv_loss.execute(outputs, target)
    print(f"   Unet DPSV Loss: {loss_val.item():.6f}")
    
    # Test PSNR_Loss
    print("\n5. Testing PSNR_Loss:")
    psnr_val = PSNR_Loss(pred, target)
    print(f"   PSNR Loss: {psnr_val.item():.6f}")
    
    # Test EPE
    print("\n6. Testing EPE (Endpoint Error):")
    epe_loss = EPE()
    flow = jt.randn(batch_size, 2, height, width)
    gt_flow = jt.randn(batch_size, 2, height, width)
    loss_mask = jt.ones(batch_size, 1, height, width)
    epe_val = epe_loss.execute(flow, gt_flow, loss_mask)
    print(f"   EPE Loss shape: {epe_val.shape}, mean: {epe_val.mean().item():.6f}")
    
    # Test Ternary
    print("\n7. Testing Ternary Loss:")
    ternary_loss = Ternary()
    loss_val = ternary_loss.execute(pred, target)
    print(f"   Ternary Loss shape: {loss_val.shape}, mean: {loss_val.mean().item():.6f}")
    
    # Test SOBEL
    print("\n8. Testing SOBEL Loss:")
    sobel_loss = SOBEL()
    loss_val = sobel_loss.execute(pred, target)
    print(f"   SOBEL Loss shape: {loss_val.shape}, mean: {loss_val.mean().item():.6f}")
    
    # Test GAN_Loss
    print("\n9. Testing GAN_Loss:")
    gan_modes = ['SGAN', 'RSGAN', 'RaSGAN', 'RaLSGAN']
    for mode in gan_modes:
        gan_loss = GAN_Loss(mode=mode)
        D_real = jt.randn(batch_size, 1)
        D_fake = jt.randn(batch_size, 1)
        D_fake_for_G = jt.randn(batch_size, 1)
        loss_D, loss_G = gan_loss.execute(D_real, D_fake, D_fake_for_G)
        print(f"   {mode:10s} - D_loss: {loss_D.item():.6f}, G_loss: {loss_G.item():.6f}")
    
    # Test gradient function
    print("\n10. Testing gradient function:")
    grad_x = gradient(pred, 'x', kernel='sobel')
    grad_y = gradient(pred, 'y', kernel='sobel')
    print(f"   Gradient X shape: {grad_x.shape}, mean: {grad_x.mean().item():.6f}")
    print(f"   Gradient Y shape: {grad_y.shape}, mean: {grad_y.mean().item():.6f}")
    
    # Test Pyramid_Sample
    print("\n11. Testing Pyramid_Sample:")
    pyramid_imgs = Pyramid_Sample(pred, max_scale=8)
    print(f"   Number of pyramid levels: {len(pyramid_imgs)}")
    for i, img in enumerate(pyramid_imgs):
        print(f"   Level {i+1}: shape={img.shape}")
    
    # Test gamma function
    print("\n12. Testing gamma function:")
    pred_positive = jt.abs(pred) + 0.1  # Ensure positive values
    gamma_out = gamma(pred_positive, clip=True, gamma_val=2.2)
    print(f"   Gamma output shape: {gamma_out.shape}, mean: {gamma_out.mean().item():.6f}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
