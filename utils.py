import numpy as np
import torch
import torch.nn.functional as F


def get_weights_and_biases(model):
    return [t for k, t in model.state_dict().items() if 'weight' in k or 'bias' in k]


def exact_interpolate(x, exact_size, scaling_factor, mode='bicubic'):
    """
    A function for performing interpolation with exact (floating point) sizes.

    Args:
        x: a 4D (N, C, H, W) torch tensor representing the image to interpolate
        exact_size: a (float, float) tuple, representing the exact computed size of the image, 
            e.g. could be (32.38, 61.77) for an input image of size (33, 62)
        scaling_factor: a float, e.g. 1.20 for upsampling
        mode: a string, upsampling mode compatible with torch.nn.interpolate's mode argument

    Returns:
        interp: the interpolated version of x
        interp_exact_size: the exact size of interp
    """
    interp_exact_size = tuple(scaling_factor * d for d in exact_size)
    interp_rounded_size = tuple(round(d) for d in exact_size)
    interp = F.interpolate(x, size=interp_rounded_size, mode=mode)
    return interp, interp_exact_size


def create_scale_pyramid(img, scaling_factor, num_scales, mode='bicubic'):
    exact_size = tuple(float(d) for d in img.shape[2:])  # (N, C, H, W) -> (H, W)
    scaled_images, exact_scale_sizes = [img], [exact_size]
    for i in range(num_scales-1):
        img, exact_size = exact_interpolate(img, exact_size, scaling_factor, mode)
        scaled_images.append(img)
        exact_scale_sizes.append(exact_size)
    return scaled_images, exact_scale_sizes


def np_image_to_normed_tensor(img_uint):
    # convert [0, 255] uint8 (H, W, C) to [-1, 1] float32 (1, C, H, W)
    rescaled = (img_uint.astype('float32') / 127.5) - 1.0
    chw = np.transpose(rescaled, (2, 0, 1))
    return torch.from_numpy(np.expand_dims(chw, axis=0))


def normed_tensor_to_np_image(img_float):
    chw = np.squeeze(img_float.detach().cpu().numpy())
    hwc = np.transpose(chw, (1, 2, 0))
    return ((hwc + 1.0) * 127.5).astype('uint8')


def gradient_penalty(discriminator, fake_batch, real_batch):

    # take a sample from the line between the real and generated images
    # for use in the gradient penalty (Impr. Training of WGANs)

    # how to calculate this loss is not very clear in this context...
    # In the case of a scalar discr. output, what should be done is simply
    # norm the gradient (image-shaped) across the channel axis, and take
    # the mean across all pixels.
    # In this case, the output of the critic (discr) is an image (PatchGAN).
    # If we take its mean to obtain a scalar and then apply the same approach
    # as the scalar output discr., it seems to suppress the penalty twice
    # (as if the mean was applied twice). Instead, taking the sum of the
    # output allows us to apply the mean only once, which we believe is the
    # proper normalization.

    batch_size = real_batch.shape[0]
    # select a random point between each real-fake pair
    epsilons = torch.rand(batch_size)
    grad_sample = epsilons * real_batch + (1 - epsilons) * fake_batch
    f_grad_sample = discriminator(grad_sample).sum()
    grad, = torch.autograd.grad(f_grad_sample, grad_sample, create_graph=True, retain_graph=True)
    grad_loss = ((torch.norm(grad, 2, dim=1) - 1) ** 2).mean()  # mean over batch
    return grad_loss
