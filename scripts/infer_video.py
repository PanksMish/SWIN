"""
infer_video.py
==============
Real-time inference on a video file — produces emotion labels per frame
and a temporal stress trajectory.

Usage:
    python scripts/infer_video.py \\
        --video path/to/video.mp4 \\
        --checkpoint checkpoints/best_model.pth \\
        --config configs/config.yaml \\
        --output results/video_output/

Output files:
    - annotated_video.mp4   : Video with emotion labels overlaid
    - stress_trajectory.png : Temporal stress curve
    - emotion_timeline.png  : Per-frame emotion probability heatmap
    - results.json          : Numerical emotion + stress data

Paper reference: Section 6 (qualitative evaluation), Figure 9–10
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

from models.full_framework import build_framework
from data.video_dataset import VideoFrameDataset, get_video_info
from data.augmentations import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from utils.stress_formulation import compute_stress_trajectory, EMOTION_CLASSES


# Emotion label colours for annotation overlay
EMOTION_COLOURS_BGR = {
    'anger':    (0,  50,  200),
    'disgust':  (140, 40, 150),
    'fear':     (200, 50,  50),
    'happiness':(50, 200,  50),
    'sadness':  (200, 100, 50),
    'surprise': (50, 150, 200),
    'neutral':  (100,100, 100),
}


@torch.no_grad()
def infer_video(
    model: torch.nn.Module,
    video_path: str,
    device: torch.device,
    sequence_length: int = 16,
    alpha: float = 0.7,
    beta: float  = 0.3,
    output_dir: str = "results/video_output",
    annotate_video: bool = True,
) -> dict:
    """
    Runs the full Swin-FANE pipeline on a video file.

    Args:
        model:           Trained Swin-FANE framework.
        video_path:      Path to input video.
        device:          Computation device.
        sequence_length: Temporal window size T.
        alpha, beta:     Stress formulation weights.
        output_dir:      Output directory for results.
        annotate_video:  Whether to produce an annotated output video.

    Returns:
        Dict with per-frame emotions and stress trajectory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # ---- Video info ----
    info = get_video_info(video_path)
    print(f"[InferVideo] Processing: {Path(video_path).name}")
    print(f"             Duration: {info['duration_sec']:.1f}s | "
          f"FPS: {info['fps']:.1f} | "
          f"Frames: {info['total_frames']}")

    # ---- Load frames ----
    print("[InferVideo] Loading and preprocessing frames...")
    dataset = VideoFrameDataset(
        video_path=video_path,
        image_size=224,
        sequence_length=sequence_length,
        use_mtcnn=True,
        skip_no_face=False,
    )

    if len(dataset) == 0:
        print("[InferVideo] No valid sequences found. Exiting.")
        return {}

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # ---- Run inference ----
    print("[InferVideo] Running emotion recognition + stress estimation...")
    all_frame_probs  = []   # (T_total, K)
    all_frame_labels = []   # (T_total,)
    all_stress_seqs  = []   # stress score per sequence

    for batch in loader:
        x_seq = batch['image'].to(device)   # (B, T, C, H, W)
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)

        output = model.forward_sequence(x_seq)

        B, T = output['frame_probs'].shape[:2]
        for b in range(B):
            probs_b  = output['frame_probs'][b].cpu().numpy()
            labels_b = output['frame_labels'][b].cpu().numpy()
            all_frame_probs.append(probs_b)
            all_frame_labels.append(labels_b)

        stress_b = output['stress_score'].cpu().numpy()
        all_stress_seqs.extend(stress_b[:, 0].tolist())

    # Flatten frame arrays
    all_frame_probs  = np.concatenate(all_frame_probs,  axis=0)  # (N_frames, K)
    all_frame_labels = np.concatenate(all_frame_labels, axis=0)  # (N_frames,)

    # ---- Compute full stress trajectory ----
    stress_info = compute_stress_trajectory(
        probs_sequence=all_frame_probs,
        alpha=alpha,
        beta=beta,
        smooth_window=7,
    )

    print(f"[InferVideo] Summary:")
    print(f"  Frames processed: {len(all_frame_labels)}")
    print(f"  Mean stress index: {stress_info['mean_stress']:.4f}")
    print(f"  Peak stress:       {stress_info['peak_stress']:.4f}")

    # Most frequent emotion
    mode_emotion = EMOTION_CLASSES[int(np.bincount(all_frame_labels).argmax())]
    print(f"  Dominant emotion:  {mode_emotion}")

    # ---- Save annotated video ----
    if annotate_video:
        _write_annotated_video(
            video_path=video_path,
            frame_labels=all_frame_labels,
            frame_probs=all_frame_probs,
            stress_traj=stress_info['smoothed'],
            output_path=os.path.join(output_dir, "annotated_video.mp4"),
        )

    # ---- Plot stress trajectory ----
    _plot_stress_trajectory(
        stress_info=stress_info,
        save_path=os.path.join(output_dir, "stress_trajectory.png"),
    )

    # ---- Plot emotion timeline heatmap ----
    _plot_emotion_timeline(
        probs=all_frame_probs,
        save_path=os.path.join(output_dir, "emotion_timeline.png"),
    )

    # ---- Save JSON results ----
    results = {
        'video':             str(video_path),
        'total_frames':      int(len(all_frame_labels)),
        'mean_stress':       float(stress_info['mean_stress']),
        'peak_stress':       float(stress_info['peak_stress']),
        'dominant_emotion':  mode_emotion,
        'stress_trajectory': stress_info['smoothed'].tolist(),
        'negative_affect':   stress_info['r'].tolist(),
        'volatility':        stress_info['v'].tolist(),
        'frame_emotions':    [EMOTION_CLASSES[int(l)] for l in all_frame_labels],
    }

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[InferVideo] Results saved to: {output_dir}")
    return results


def _write_annotated_video(
    video_path: str,
    frame_labels: np.ndarray,
    frame_probs: np.ndarray,
    stress_traj: np.ndarray,
    output_path: str,
):
    """Writes an annotated video with emotion labels and stress bar."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(frame_labels):
            break

        emotion_name = EMOTION_CLASSES[int(frame_labels[frame_idx])]
        colour       = EMOTION_COLOURS_BGR.get(emotion_name, (200, 200, 200))
        stress_val   = float(stress_traj[min(frame_idx, len(stress_traj) - 1)])
        confidence   = float(frame_probs[frame_idx].max())

        # Emotion label box
        cv2.rectangle(frame, (10, 10), (280, 70), colour, -1)
        cv2.putText(frame, emotion_name.upper(), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Confidence score
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Stress bar (bottom of frame)
        bar_w = int(w * stress_val)
        stress_colour = (0, int(255 * (1 - stress_val)), int(255 * stress_val))
        cv2.rectangle(frame, (0, h - 20), (bar_w, h), stress_colour, -1)
        cv2.putText(frame, f"Stress: {stress_val:.3f}", (10, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[InferVideo] Annotated video saved: {output_path}")


def _plot_stress_trajectory(stress_info: dict, save_path: str):
    """Plots the temporal stress trajectory with annotations."""
    T     = len(stress_info['smoothed'])
    time  = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Plot 1: Stress trajectory
    ax = axes[0]
    ax.fill_between(time, stress_info['smoothed'], alpha=0.3, color='#e74c3c')
    ax.plot(time, stress_info['smoothed'], color='#c0392b', linewidth=2)
    ax.axhline(y=stress_info['mean_stress'], color='gray', linestyle='--',
               label=f"Mean = {stress_info['mean_stress']:.3f}")
    ax.set_ylabel('Stress Score S(t)', fontsize=10)
    ax.set_title('Temporal Stress Trajectory', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Negative affect
    ax2 = axes[1]
    ax2.fill_between(time, stress_info['r'], alpha=0.3, color='#3498db')
    ax2.plot(time, stress_info['r'], color='#2980b9', linewidth=1.5)
    ax2.set_ylabel('Negative Affect r(t)', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Volatility
    ax3 = axes[2]
    ax3.fill_between(time, stress_info['v'], alpha=0.3, color='#e67e22')
    ax3.plot(time, stress_info['v'], color='#d35400', linewidth=1.5)
    ax3.set_ylabel('Volatility v(t)', fontsize=10)
    ax3.set_xlabel('Frame Index', fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_emotion_timeline(probs: np.ndarray, save_path: str):
    """Plots per-frame emotion probability heatmap."""
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(probs.T, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1, origin='lower')

    ax.set_yticks(range(len(EMOTION_CLASSES)))
    ax.set_yticklabels([c.capitalize() for c in EMOTION_CLASSES])
    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_title('Per-Frame Emotion Probability Timeline', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Probability')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Swin-FANE on video')
    parser.add_argument('--video',       type=str, required=True)
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--config',      type=str, required=True)
    parser.add_argument('--output',      type=str, default='results/video_output')
    parser.add_argument('--no-annotate', action='store_true',
                        help='Skip writing annotated video')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_framework(config).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])

    infer_video(
        model=model,
        video_path=args.video,
        device=device,
        sequence_length=config['dataset']['sequence_length'],
        alpha=config['stress']['alpha'],
        beta=config['stress']['beta'],
        output_dir=args.output,
        annotate_video=not args.no_annotate,
    )
