import torch
import torch.nn as nn

try:
    from .encoders import init_raft_small, init_dino_v2, encode_dino_v2
    from .video_io import VideoReader
except ImportError:
    from encoders import init_raft_small, init_dino_v2, encode_dino_v2
    from video_io import VideoReader

from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin
import matplotlib.pyplot as plt
import numpy as np
import pywt
import ruptures as rpt

class SimilarityCurves(nn.Module):
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.raft_model, self.raft_preprocess = init_raft_small(device=device)
        self.dino_model, self.dino_preprocess = init_dino_v2(device=device)

    def _extract_features(self, video_path, fps = 1.0, batch_size = 32):
        vr = VideoReader(video_path)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        sample_interval = int(video_fps / fps)
        frame_indices = list(range(0, total_frames, sample_interval))

        raw_frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(frame) for frame in raw_frames]

        all_dino = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            all_dino.append(encode_dino_v2(self.dino_model, self.dino_preprocess, batch, device=self.device).cpu())
        dino_features = torch.cat(all_dino, dim=0)

        all_flow = []
        for i in range(len(frames) - 1):
            t1, t2 = self.raft_preprocess(frames[i], frames[i + 1])
            t1 = t1.unsqueeze(0).to(self.device)
            t2 = t2.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                flow = self.raft_model(t1, t2)
            all_flow.append(flow[-1].cpu())
        raft_features = torch.stack(all_flow)

        return raft_features, dino_features

    def compute_scene_similarity(self, dino_features):
        scene_similarity = F.cosine_similarity(dino_features[1:], dino_features[:-1], dim=1)
        scene_similarity = torch.cat([torch.tensor([1.0]), scene_similarity])
        return scene_similarity

    def compute_action_similarity(self, raft_features):
        magnitude = torch.norm(raft_features.view(raft_features.size(0), -1), dim=1)
        action_similarity = -magnitude
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

    def compute_scene(self, video_path, fps=1.0, batch_size=32, tau=None):
        vr = VideoReader(video_path)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        sample_interval = int(video_fps / fps)
        frame_indices = list(range(0, total_frames, sample_interval))

        raw_frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(f) for f in raw_frames]

        all_dino = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            all_dino.append(encode_dino_v2(self.dino_model, self.dino_preprocess, batch, device=self.device).cpu())
        dino_features = torch.cat(all_dino, dim=0)

        scene_similarity = self.compute_scene_similarity(dino_features)
        scene_curve = self.smooth(scene_similarity)
        scene_bps = self.scene_boundaries(scene_curve, tau=tau)

        return {
            'curve': scene_curve,
            'boundaries': scene_bps,
            'frame_indices': frame_indices,
            'num_sampled_frames': len(frame_indices),
        }

    def compute_action_for_segments(self, video_path, fps=1.0, segments=None, pen=0.5, model="rbf"):
        vr = VideoReader(video_path)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        sample_interval = int(video_fps / fps)
        all_frame_indices = list(range(0, total_frames, sample_interval))

        results = []
        for (start, end) in segments:
            start = max(0, start)
            end = min(len(all_frame_indices) - 1, end)

            seg_frame_indices = all_frame_indices[start:end + 1]
            raw_frames = vr.get_batch(seg_frame_indices).asnumpy()
            frames = [Image.fromarray(f) for f in raw_frames]

            if len(frames) < 2:
                results.append({
                    'curve': np.array([]),
                    'local_boundaries': [],
                    'global_boundaries': [],
                    'segment': (start, end),
                })
                continue

            all_flow = []
            for i in range(len(frames) - 1):
                t1, t2 = self.raft_preprocess(frames[i], frames[i + 1])
                t1 = t1.unsqueeze(0).to(self.device)
                t2 = t2.unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    flow = self.raft_model(t1, t2)
                all_flow.append(flow[-1].cpu())
            raft_features = torch.stack(all_flow)

            action_sim = self.compute_action_similarity(raft_features)
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

    def compute_all(self, video_path, fps=1.0, batch_size=32):
        raft_features, dino_features = self._extract_features(video_path, fps=fps, batch_size=batch_size)
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
