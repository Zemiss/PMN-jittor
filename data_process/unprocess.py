from utils import fn_timer
import numpy as np
import jittor as jt
# from utils import fn_timer


def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # # Takes a random convex combination of XYZ -> Camera CCMs.
    # xyz2cams = [[[1.0234, -0.2969, -0.2266],
    #            [-0.5625, 1.6328, -0.0469],
    #            [-0.0703, 0.2188, 0.6406]],
    #           [[0.4913, -0.0541, -0.0202],
    #            [-0.613, 1.3513, 0.2906],
    #            [-0.1564, 0.2151, 0.7183]],
    #           [[0.838, -0.263, -0.0639],
    #            [-0.2887, 1.0725, 0.2496],
    #            [-0.0627, 0.1427, 0.5438]],
    #           [[0.6596, -0.2079, -0.0562],
    #            [-0.4782, 1.3016, 0.1933],
    #            [-0.097, 0.1581, 0.5181]]]
    # num_ccms = len(xyz2cams)
    # xyz2cams = torch.FloatTensor(xyz2cams)
    # weights  = torch.FloatTensor(num_ccms, 1, 1).uniform_(1e-8, 1e8)
    # weights_sum = torch.sum(weights, dim=0)
    # xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

    # # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    # rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
    #                            [0.2126729, 0.7151522, 0.0721750],
    #                            [0.0193339, 0.1191920, 0.9503041]])
    # rgb2cam = torch.mm(xyz2cam, rgb2xyz)
    # SonyA7S2 ccm's inv
    rgb2cam = [[1.,0.,0.],
               [0.,1.,0.],
               [0.,0.,1.]]
    # # RedMi K30 ccm's inv
    # rgb2cam = [[0.61093086,0.31565922,0.07340994],
    #             [0.09433191,0.7658969,0.1397712 ],
    #             [0.03532438,0.3020709,0.6626047 ]]
    rgb2cam = jt.array(rgb2cam).float32()
    # # Normalizes each row.
    # rgb2cam = rgb2cam / jt.sum(rgb2cam, dim=-1, keepdim=True)
    return rgb2cam


#def random_gains():
#    """Generates random gains for brightening and white balance."""
#    # RGB gain represents brightening.
#    n        = tdist.Normal(loc=torch.tensor([0.8]), scale=torch.tensor([0.1]))
#    rgb_gain = 1.0 / n.sample()
#
#    # Red and blue gains represent white balance.
#    red_gain  =  torch.FloatTensor(1).uniform_(1.9, 2.4)
#    blue_gain =  torch.FloatTensor(1).uniform_(1.5, 1.9)
#    return rgb_gain, red_gain, blue_gain

def random_gains():
    # return jt.array(np.array([[1.],[1.],[1.]])).float32()
    # Use numpy for normal distribution sampling since jittor doesn't have tdist
    rgb_gain = 1.0 / np.random.normal(0.8, 0.1)
    red_gain = np.random.uniform(1.4, 2.3)
    ployfit = [6.14381188, -3.65620261, 0.70205967]
    blue_gain= ployfit[0] + ployfit[1] * red_gain + ployfit[2] * red_gain ** 2# + np.random.uniform(0, 0.4)
    red_gain = jt.array([red_gain]).float32().view(1)
    blue_gain = jt.array([blue_gain]).float32().view(1)
    rgb_gain = jt.array([rgb_gain]).float32()
    return rgb_gain, red_gain, blue_gain

def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    #image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    image = jt.clamp(image, min_v=0.0, max_v=1.0)
    out   = 0.5 - jt.sin(jt.asin(1.0 - 2.0 * image) / 3.0)
    #out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    #image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    out   = jt.clamp(image, min_v=1e-8) ** 2.2
    #out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    shape = image.size()
    image = jt.reshape(image, [-1, 3])
    image = jt.matmul(image, ccm)
    out   = jt.reshape(image, shape)
    return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain, use_gpu=False):
    """Inverts gains while safely handling saturated pixels."""
    # H, W, C
    green = jt.array([1.0]).float32()
    gains = jt.stack((1.0 / red_gain, green, 1.0 / blue_gain)) / rgb_gain
    gains = gains.view(1,1,3)
    #gains = gains[None, None, :]
    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray  = jt.mean(image, dim=-1, keepdims=True)
    inflection = 0.9
    mask  = (jt.clamp(gray - inflection, min_v=0.0) / (1.0 - inflection)) ** 2.0

    safe_gains = jt.maximum(mask + (1.0 - mask) * gains, gains)
    out   = image * safe_gains
    return out

def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    if image.size() == 3:
        image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
        shape = image.size()
        red   = image[0::2, 0::2, 0]
        green_red  = image[0::2, 1::2, 1]
        green_blue = image[1::2, 0::2, 1]
        blue = image[1::2, 1::2, 2]
        out  = jt.stack((red, green_red, blue, green_blue), dim=-1)
        out  = jt.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
        out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    else: # [crops, t, h, w, c]
        shape = image.size()
        red   = image[..., 0::2, 0::2, 0]
        green_red  = image[..., 0::2, 1::2, 1]
        green_blue = image[..., 1::2, 0::2, 1]
        blue = image[..., 1::2, 1::2, 2]
        out  = jt.stack((red, green_red, blue, green_blue), dim=-1)
        # out  = jt.reshape(out, (shape[0], shape[1], shape[-3] // 2, shape[-2] // 2, 4))
        # out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out

def mosaic_GBRG(image):
    """Extracts GBRG Bayer planes from an RGB image."""
    if image.size() == 3:
        image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
        shape = image.size()
        red   = image[1::2, 0::2, 0]
        green_red  = image[1::2, 1::2, 1]
        green_blue = image[0::2, 0::2, 1]
        blue = image[0::2, 1::2, 2]
        out  = jt.stack((red, green_red, green_blue, blue), dim=-1)
        out  = jt.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
        out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    else: # [crops, t, h, w, c]
        shape = image.size()
        red   = image[..., 1::2, 0::2, 0]
        green_red  = image[..., 1::2, 1::2, 1]
        green_blue = image[..., 0::2, 0::2, 1]
        blue = image[..., 0::2, 1::2, 2]
        out  = jt.stack((red, green_red, green_blue, blue), dim=-1)
        # out  = jt.reshape(out, (shape[0], shape[1], shape[-3] // 2, shape[-2] // 2, 4))
        # out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out

# @ fn_timer
def unprocess(image, lock_wb=False, use_gpu=False):
    """Unprocesses an image from sRGB to realistic raw data."""
    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = jt.linalg.inv(rgb2cam)
    # rgb_gain, red_gain, blue_gain = random_gains() if lock_wb is False else jt.array(np.array([[1.],[2.],[2.]])).float32()
    rgb_gain, red_gain, blue_gain = random_gains() if lock_wb is False else jt.array(np.array(lock_wb)).float32()
    if len(image.size()) >= 4:
        res = image.clone()
        for i in range(image.size()[0]):
            temp = image[i]
            temp = inverse_smoothstep(temp)
            temp = gamma_expansion(temp)
            temp = apply_ccm(temp)
            temp = safe_invert_gains(temp, rgb_gain, red_gain, blue_gain, use_gpu)
            temp = jt.clamp(temp, min_v=0.0, max_v=1.0)
            res[i]= temp.clone()
        
        metadata = {
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
        }
        return res, metadata
    else:
        # Approximately inverts global tone mapping.
        image = inverse_smoothstep(image)
        # Inverts gamma compression.
        image = gamma_expansion(image)
        # Inverts color correction.
        image = apply_ccm(image, rgb2cam)
        # Approximately inverts white balance and brightening.
        image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain, use_gpu)
        # Clips saturated pixels.
        image = jt.clamp(image, min_v=0.0, max_v=1.0)
        # Applies a Bayer mosaic.
        #image = mosaic(image)

        metadata = {
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
        }
        return image, metadata


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(0.0, 0.26)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    image    = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    variance = image * shot_noise + read_noise
    noise = jt.init.gauss(variance.shape, 'float32') * jt.sqrt(variance)
    out      = image + noise
    out      = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out

if __name__ == '__main__':
    # Test with numpy poisson distribution
    for i in range(10):
        s = np.random.poisson([10., 100., 1000.])
        print(s)