import math
import torch
import torch.nn.functional as F

GRAY_WEIGHTS = (0.299, 0.587, 0.114)
JITTER = 0.2


def augment(images, config):
    images = random_resized_crop_and_flip(images, config.min_crop_area, config.image_size)
    images = colour_jitter(images)
    images = random_grayscale(images, config.grayscale_probability)
    images = random_erase(images, config.erase_probability, config.erase_area)
    return images.contiguous(memory_format=torch.channels_last)


def random_resized_crop_and_flip(images, min_area, size):
    batch, device = len(images), images.device
    area = torch.empty(batch, device=device).uniform_(min_area, 1.0)
    aspect = torch.exp(torch.empty(batch, device=device).uniform_(math.log(3 / 4), math.log(4 / 3)))
    scale_x = (area * aspect).sqrt().clamp(max=1.0)
    scale_y = (area / aspect).sqrt().clamp(max=1.0)
    flip = torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = scale_x * flip
    theta[:, 1, 1] = scale_y
    theta[:, 0, 2] = (torch.rand(batch, device=device) * 2 - 1) * (1 - scale_x)
    theta[:, 1, 2] = (torch.rand(batch, device=device) * 2 - 1) * (1 - scale_y)
    grid = F.affine_grid(theta, (batch, 3, size, size), align_corners=False)
    return F.grid_sample(images.float(), grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def grayscale(images):
    return (images * images.new_tensor(GRAY_WEIGHTS).view(1, 3, 1, 1)).sum(1, keepdim=True)


def colour_jitter(images):
    brightness, contrast, saturation = torch.empty(3, len(images), 1, 1, 1, device=images.device).uniform_(1 - JITTER, 1 + JITTER)
    images = images * brightness
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * contrast + mean
    gray = grayscale(images)
    return ((images - gray) * saturation + gray).clamp(0, 255)


def random_grayscale(images, probability):
    chosen = torch.rand(len(images), 1, 1, 1, device=images.device) < probability
    return torch.where(chosen, grayscale(images).expand_as(images), images)


def random_boxes(heights, widths, size):
    batch, device = len(heights), heights.device
    top = (torch.rand(batch, device=device) * (size - heights))[:, None, None]
    left = (torch.rand(batch, device=device) * (size - widths))[:, None, None]
    rows = torch.arange(size, device=device)[None, :, None]
    columns = torch.arange(size, device=device)[None, None, :]
    return (rows >= top) & (rows < top + heights[:, None, None]) & (columns >= left) & (columns < left + widths[:, None, None])


def random_erase(images, probability, area_range):
    batch, size, device = len(images), images.shape[-1], images.device
    area = torch.empty(batch, device=device).uniform_(*area_range)
    aspect = torch.exp(torch.empty(batch, device=device).uniform_(math.log(1 / 3), math.log(3)))
    heights = (area / aspect).sqrt().clamp(max=1.0) * size
    widths = (area * aspect).sqrt().clamp(max=1.0) * size
    erase = random_boxes(heights, widths, size) & (torch.rand(batch, 1, 1, device=device) < probability)
    return torch.where(erase[:, None], images.mean(dim=(2, 3), keepdim=True), images)

