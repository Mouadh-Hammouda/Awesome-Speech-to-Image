import torch.nn.functional as F


def info_nce(images, captions, caption_images, logit_scale):
    # Each caption matches its one image; each image matches all of its captions in the batch, weighted equally.
    scores = logit_scale.exp() * captions @ images.T
    matches = F.one_hot(caption_images, len(images)).float().T
    caption_to_image = F.cross_entropy(scores, caption_images)
    image_to_caption = -(matches / matches.sum(1, keepdim=True) * scores.T.log_softmax(dim=1)).sum(1).mean()
    return (caption_to_image + image_to_caption) / 2
