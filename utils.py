import torch
import random
import numpy as np
from fr_model import IRSE_50, MobileFaceNet, IR_152, InceptionResnetV1
import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_fr_model(name, device=torch.device("cuda")):
    if name == 'IRSE50':
        model = IRSE_50()
        model.load_state_dict(torch.load('pretrained_model/irse50.pth', map_location="cpu"))
    elif name == 'MobileFace':
        model = MobileFaceNet(512)
        model.load_state_dict(torch.load('pretrained_model/mobile_face.pth', map_location="cpu"))
    elif name == 'IR152':
        model = IR_152([112, 112])
        model.load_state_dict(torch.load('pretrained_model/ir152.pth', map_location="cpu"))
    elif name == 'FaceNet':
        model = InceptionResnetV1(num_classes=8631)
        model.load_state_dict(torch.load('pretrained_model/facenet.pth', map_location="cpu"))
    else:
        raise ValueError(f'Invalid model name: {name}')
    return model.to(device)


def compute_fr_losses(decoded, src, tgt, attack_model_dict, size=112):
    # Resize to FR input size
    def resize_for_fr(x):
        return torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    decoded_resized = resize_for_fr(decoded)
    src_resized = resize_for_fr(src)
    tgt_resized = resize_for_fr(tgt)

    L_fr_tgt_total, L_fr_src_total = 0.0, 0.0
    for name, fr_model in attack_model_dict.items():
        e_pred = fr_model(decoded_resized)
        e_src  = fr_model(src_resized)
        e_tgt  = fr_model(tgt_resized)

        # Normalize embeddings
        e_pred = F.normalize(e_pred, p=2, dim=1)
        e_src  = F.normalize(e_src,  p=2, dim=1)
        e_tgt  = F.normalize(e_tgt,  p=2, dim=1)

        # Cosine similarities
        cos_pred_src = (e_pred * e_src).sum(dim=1)
        cos_pred_tgt = (e_pred * e_tgt).sum(dim=1)

        # Losses: preserve source, fool toward target
        L_fr_src_total += (1 - cos_pred_src).mean()
        L_fr_tgt_total += (1 - cos_pred_tgt).mean()

    return L_fr_src_total / len(attack_model_dict), L_fr_tgt_total / len(attack_model_dict)
