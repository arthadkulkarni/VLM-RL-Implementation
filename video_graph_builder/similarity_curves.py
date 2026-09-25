import torch
import torch.nn as nn

try:
    from .encoders import init_raft_small, init_dino_v2, encode_dino_v2
except ImportError:
    from encoders import init_raft_small, init_dino_v2, encode_dino_v2

from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin
import matplotlib.pyplot as plt
import numpy as np
import pywt
import ruptures as rpt

# RAFT input width; see SimilarityCurves.flow_magnitudes.
RAFT_WIDTH = 480

class SimilarityCurves(nn.Module):
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.raft_model, self.raft_preprocess = init_raft_small(device=device)
        self.dino_model, self.dino_preprocess = init_dino_v2(device=device)

    @torch.inference_mode()
    def dino_features(self, frames, batch_size=64):
        """DINOv2 CLS embedding per sampled frame (drives scene cuts)."""
        return torch.cat([
            encode_dino_v2(self.dino_model, self.dino_preprocess, frames[i:i + batch_size], device=self.device).cpu()
            for i in range(0, len(frames), batch_size)
        ], dim=0)

    @torch.inference_mode()
    def flow_magnitudes(self, frames, batch_size=32):
        """L2 norm of RAFT flow for every consecutive pair of sampled frames
        (entry k is frames[k] -> frames[k+1]), batched. Frames are resized to
        RAFT_WIDTH (dims rounded to multiples of 8, which RAFT requires); the
        action curve is min-max normalized per segment, so only relative
        magnitude matters. Only the norm is kept -- full flow fields for a
        long video would not fit in memory."""
        if len(frames) < 2:
            return torch.empty(0)
        h, w = frames[0].shape[:2]
        scale = min(1.0, RAFT_WIDTH / w)
        size = (max(8, round(h * scale / 8) * 8), max(8, round(w * scale / 8) * 8))

        mags = []
        for i in range(0, len(frames) - 1, batch_size):
            chunk = torch.from_numpy(np.stack(frames[i:i + batch_size + 1])).to(self.device)
            chunk = chunk.permute(0, 3, 1, 2).float()
            chunk = F.interpolate(chunk, size=size, mode="bilinear", antialias=True, align_corners=False)
            chunk = chunk / 127.5 - 1.0  # RAFT preset: [0, 255] -> [-1, 1]
            flow = self.raft_model(chunk[:-1].contiguous(), chunk[1:].contiguous())[-1]
            mags.append(flow.flatten(1).norm(dim=1).cpu())
        return torch.cat(mags)

    def _extract_features(self, sampled, batch_size=32):
        return self.flow_magnitudes(sampled.frames), self.dino_features(sampled.frames, batch_size=batch_size)

    def compute_scene_similarity(self, dino_features):
        scene_similarity = F.cosine_similarity(dino_features[1:], dino_features[:-1], dim=1)
        scene_similarity = torch.cat([torch.tensor([1.0]), scene_similarity])
        return scene_similarity

    def compute_action_similarity(self, flow_magnitudes):
        action_similarity = -flow_magnitudes
        action_similarity = torch.cat([torch.tensor([0.0]), action_similarity])
        denom = action_similarity.max() - action_similarity.min()
        action_similarity = (action_similarity - action_similarity.min()) / (denom + 1e-8)
        return action_similarity

    def smooth(self, signal, threshold_scale=0.5, mode='soft', level_offset=3):
        curve = np.array(signal)
        N = len(curve)
        J = max(0, int(np.floor(np.log2(N))) - level_offset)
        coeffs = pywt.wavedec(curve, 'db4', level=J)
        sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma_est * np.sqrt(2 * np.log(N)) * threshold_scale
        coeffs[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
        smoothed_curve = pywt.waverec(coeffs, 'db4')
        return smoothed_curve[:N]

    def smooth_gaussian(self, signal, sigma=2.0):
        curve = np.array(signal)
        return gaussian_filter1d(curve, sigma=sigma)

    def compute_scene(self, sampled, batch_size=64, tau=None):
        dino_features = self.dino_features(sampled.frames, batch_size=batch_size)
        scene_similarity = self.compute_scene_similarity(dino_features)
        scene_curve = self.smooth(scene_similarity)
        scene_bps = self.scene_boundaries(scene_curve, tau=tau)

        return {
            'curve': scene_curve,
            'boundaries': scene_bps,
            'frame_indices': sampled.frame_indices,
            'num_sampled_frames': len(sampled.frame_indices),
        }

    def compute_action_for_segments(self, sampled, segments, pen=0.5, model="rbf"):
        # One batched RAFT pass over the whole video; each segment's curve is
        # its slice (a segment's frames start..end give pairs start..end-1).
        magnitudes = self.flow_magnitudes(sampled.frames)
        last_idx = len(sampled.frame_indices) - 1

        results = []
        for (start, end) in segments:
            start = max(0, start)
            end = min(last_idx, end)

            if end - start < 1:
                results.append({
                    'curve': np.array([]),
                    'local_boundaries': [],
                    'global_boundaries': [],
                    'segment': (start, end),
                })
                continue

            action_sim = self.compute_action_similarity(magnitudes[start:end])
            action_curve = self.smooth(action_sim, threshold_scale=0.2, mode='hard')
            local_bps = self.action_boundaries(action_curve, pen=pen, model=model)
            global_bps = [start + bp for bp in local_bps]

            results.append({
                'curve': action_curve,
                'local_boundaries': local_bps,
                'global_boundaries': global_bps,
                'segment': (start, end),
            })

        return results

    def compute_all(self, sampled, batch_size=32):
        raft_features, dino_features = self._extract_features(sampled, batch_size=batch_size)
        scene_similarity = self.compute_scene_similarity(dino_features)
        action_similarity = self.compute_action_similarity(raft_features)

        scene_similarity_smooth = self.smooth(scene_similarity)
        action_similarity_smooth = self.smooth(action_similarity, threshold_scale=0.2, mode='hard')

        similarity_dict = {
            'scene': scene_similarity_smooth,
            'action': action_similarity_smooth,
        }
        return similarity_dict

    def scene_boundaries(self, curve, tau=None, order=2):
        if tau is None:
            tau = curve.mean()
        minima = argrelmin(curve, order=order)[0]
        return [int(i) for i in minima if curve[i] < tau]

    def action_boundaries_fine(self, curve, tau=None, order=1):
        if tau is None:
            tau = curve.mean()
        minima = argrelmin(curve, order=order)[0]
        return [int(i) for i in minima if curve[i] < tau]

    def action_boundaries(self, curve, pen=0.5, model="rbf"):
        bps = rpt.Pelt(model=model).fit(curve.reshape(-1, 1)).predict(pen=pen)
        return bps[:-1]

    def compute_boundaries(self, similarity_dict, tau=None, pen=0.5, model="rbf"):
        scene_curve = similarity_dict['scene']
        action_curve = similarity_dict['action']
        scene_bps = self.scene_boundaries(scene_curve, tau=tau)
        action_bps = self.action_boundaries(action_curve, pen=pen, model=model)
        return {
            'scene': scene_bps,
            'action': action_bps,
        }

    def timestamp_to_seconds(self, timestamp: str):
        h, m, s = map(float, timestamp.split(':'))
        return h * 3600 + m * 60 + s

    def seconds_to_timestamp(self, seconds: float):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def apply_offsets(self, segments, offsets):
        adjusted_segments = []
        for (start, end), (offset_start, offset_end) in zip(segments, offsets):
            adjusted_start = max(0, start + offset_start)
            adjusted_end = max(adjusted_start + 1, end + offset_end)
            adjusted_segments.append((adjusted_start, adjusted_end))
        return adjusted_segments

    def plot_similarity_curves(self, similarity_dict, boundaries_dict=None):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axes[0].plot(similarity_dict['scene'], color='blue')
        axes[0].set_title('Scene Similarity')
        axes[0].set_ylabel('Similarity')

        axes[1].plot(similarity_dict['action'], color='red')
        axes[1].set_title('Action Similarity')
        axes[1].set_ylabel('Similarity')
        axes[1].set_xlabel('Frame Index')

        if boundaries_dict is not None:
            for b in boundaries_dict.get('scene', []):
                axes[0].axvline(x=b, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
            for b in boundaries_dict.get('action', []):
                axes[1].axvline(x=b, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

        plt.tight_layout()
        plt.show()
