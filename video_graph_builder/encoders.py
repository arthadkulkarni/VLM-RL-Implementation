import numpy as np
import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from transformers import AutoImageProcessor, AutoModel


def init_raft_small(device: str = 'cuda'):
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device).eval()
    transforms = weights.transforms()

    def raft_preprocess(frame1, frame2):
        t1 = torch.from_numpy(np.array(frame1)).permute(2, 0, 1).unsqueeze(0)
        t2 = torch.from_numpy(np.array(frame2)).permute(2, 0, 1).unsqueeze(0)
        t1, t2 = transforms(t1, t2)
        return t1.squeeze(0), t2.squeeze(0)

    return model, raft_preprocess


def init_dino_v2(device: str = 'cuda', model_name: str = 'facebook/dinov2-base'):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def dino_preprocess(images):
        return processor(images=images, return_tensors='pt')

    return model, dino_preprocess


@torch.inference_mode()
def encode_dino_v2(model, preprocess, frames, device: str = 'cuda'):
    inputs = preprocess(frames)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]
