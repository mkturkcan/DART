"""ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022.

Key idea: use ALL detection boxes (not just high-confidence ones) for
association. High-score detections are matched first via IoU to existing
tracks. Then low-score detections get a second chance to match unmatched
tracks, recovering occluded objects that other trackers would lose.

This implementation is self-contained with no external tracker dependencies.
Kalman predict/update and IoU computations are fully vectorized with numpy.
"""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Vectorized Kalman filter (constant-velocity model for bounding boxes)
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Kalman filter for axis-aligned bounding box tracking.

    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement:  [cx, cy, w, h]

    Uses constant-velocity motion model with Gaussian noise.
    All operations support both single-track and batched (N tracks) inputs.
    """

    def __init__(self):
        dt = 1.0  # time step (1 frame)

        # State transition matrix F (8x8): pos += vel * dt
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Precompute F transpose for reuse
        self.FT = self.F.T.copy()

        # Measurement matrix H (4x8): observe position only
        self.H = np.eye(4, 8, dtype=np.float64)
        self.HT = self.H.T.copy()

        # Noise scale factors
        self._std_weight_position = 1.0 / 15
        self._std_weight_velocity = 1.0 / 40

        # Precomputed identity matrices
        self._I8 = np.eye(8, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create initial state from measurement [cx, cy, w, h]."""
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = measurement

        w, h = measurement[2], measurement[3]
        p = self._std_weight_position
        v = self._std_weight_velocity
        std = np.array([
            2 * p * w, 2 * p * h, 2 * p * w, 2 * p * h,
            10 * v * w, 10 * v * h, 10 * v * w, 10 * v * h,
        ])
        covariance = np.diag(std * std)
        return mean, covariance

    def predict_batch(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state for N tracks.

        Args:
            means: (N, 8) state vectors.
            covariances: (N, 8, 8) covariance matrices.

        Returns:
            (new_means, new_covariances) with same shapes.
        """
        p = self._std_weight_position
        v = self._std_weight_velocity
        ws = means[:, 2]  # (N,)
        hs = means[:, 3]  # (N,)

        # Process noise Q: diagonal (N, 8, 8)
        stds = np.column_stack([
            p * ws, p * hs, p * ws, p * hs,
            v * ws, v * hs, v * ws, v * hs,
        ])  # (N, 8)
        Q = np.zeros((len(means), 8, 8), dtype=np.float64)
        diag_idx = np.arange(8)
        Q[:, diag_idx, diag_idx] = stds * stds

        # Batch predict: mean_new = mean @ F.T, cov_new = F @ cov @ F.T + Q
        new_means = means @ self.FT           # (N, 8)
        temp = np.matmul(self.F, covariances)  # (N, 8, 8)
        new_covs = np.matmul(temp, self.FT) + Q

        return new_means, new_covs

    def update_batch(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        measurements: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman update for M matched tracks.

        Args:
            means: (M, 8) predicted state vectors.
            covariances: (M, 8, 8) predicted covariance matrices.
            measurements: (M, 4) measurement vectors [cx, cy, w, h].

        Returns:
            (new_means, new_covariances) with same shapes.
        """
        M = means.shape[0]
        p = self._std_weight_position
        ws = means[:, 2]
        hs = means[:, 3]

        # Measurement noise R: diagonal (M, 4, 4)
        stds = np.column_stack([p * ws, p * hs, p * ws, p * hs])
        R = np.zeros((M, 4, 4), dtype=np.float64)
        diag4 = np.arange(4)
        R[:, diag4, diag4] = stds * stds

        # Projected state
        projected_mean = means[:, :4]  # H @ mean = mean[:4]
        # projected_cov = H @ cov @ H.T + R
        projected_cov = covariances[:, :4, :4] + R  # (M, 4, 4)

        # Kalman gain: K = cov @ H.T @ inv(projected_cov)
        # Using solve: projected_cov @ X = (cov @ H.T).T  =>  X.T = K
        cov_HT = covariances[:, :, :4]  # (M, 8, 4) — cov @ H.T
        # Solve projected_cov @ X = cov_HT^T for each track
        # X shape: (M, 4, 8), K = X^T -> (M, 8, 4)
        X = np.linalg.solve(
            projected_cov,
            cov_HT.transpose(0, 2, 1),  # (M, 4, 8)
        )  # (M, 4, 8)
        K = X.transpose(0, 2, 1)  # (M, 8, 4)

        # Innovation
        innovation = measurements - projected_mean  # (M, 4)
        new_means = means + np.einsum("nij,nj->ni", K, innovation)

        # Joseph form is more stable but standard form is fine here
        # new_cov = (I - K @ H) @ cov
        KH = np.matmul(K, np.broadcast_to(self.H, (M, 4, 8)))  # (M, 8, 8)
        I_KH = self._I8 - KH  # (M, 8, 8) via broadcast
        new_covs = np.matmul(I_KH, covariances)

        return new_means, new_covs


# ---------------------------------------------------------------------------
# Single track
# ---------------------------------------------------------------------------

class STrack:
    """Represents a single tracked object."""

    _next_id = 1
    _score_ema_alpha = 0.7

    def __init__(self, box_xyxy: np.ndarray, score: float, class_id: int):
        self.score = score
        self.class_id = class_id

        # Class smoothing: exponential-decay score-weighted votes per class.
        self.class_scores = {class_id: score}
        self._class_decay = 0.85

        # Kalman state (initialized externally via set_state)
        cx = (box_xyxy[0] + box_xyxy[2]) * 0.5
        cy = (box_xyxy[1] + box_xyxy[3]) * 0.5
        w = box_xyxy[2] - box_xyxy[0]
        h = box_xyxy[3] - box_xyxy[1]
        self._measurement = np.array([cx, cy, w, h], dtype=np.float64)

        self.mean = None       # set by tracker after initiate
        self.covariance = None

        self.track_id = 0
        self.is_activated = False
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0

    @property
    def box_xyxy(self) -> np.ndarray:
        """Current bounding box in [x1, y1, x2, y2] format."""
        cx, cy, w, h = self.mean[0], self.mean[1], self.mean[2], self.mean[3]
        return np.array([
            cx - w * 0.5, cy - h * 0.5,
            cx + w * 0.5, cy + h * 0.5,
        ], dtype=np.float32)

    def activate(self, frame_id: int) -> None:
        self.track_id = STrack._next_id
        STrack._next_id += 1
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0

    def _update_class(self, det_class: int, det_score: float) -> None:
        for k in self.class_scores:
            self.class_scores[k] *= self._class_decay
        self.class_scores[det_class] = (
            self.class_scores.get(det_class, 0.0) + det_score
        )
        self.class_id = max(self.class_scores, key=self.class_scores.get)

    def apply_update(
        self, det_box: np.ndarray, det_score: float, det_class: int, frame_id: int
    ) -> None:
        """Apply detection update (Kalman state set externally by tracker)."""
        alpha = self._score_ema_alpha
        self.score = alpha * det_score + (1.0 - alpha) * self.score
        self._update_class(det_class, det_score)
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.is_activated = True

    @staticmethod
    def reset_id():
        STrack._next_id = 1


# ---------------------------------------------------------------------------
# IoU and NMS
# ---------------------------------------------------------------------------

def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of xyxy boxes. (M,4) x (N,4) -> (M,N)."""
    M, N = boxes_a.shape[0], boxes_b.shape[0]
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def nms_class_agnostic(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Greedy class-agnostic NMS. Returns indices to keep.

    Unlike per-class NMS, this suppresses overlapping boxes regardless
    of class. Useful for similar classes (car/suv/van) that should not
    coexist on the same spatial location.

    Args:
        boxes: (N, 4) xyxy boxes.
        scores: (N,) detection scores.
        class_ids: (N,) class IDs (unused, kept for API symmetry).
        iou_threshold: IoU above which to suppress.

    Returns:
        (K,) integer array of kept indices.
    """
    if len(scores) == 0:
        return np.empty(0, dtype=np.intp)

    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        # Compute IoU of box i against all remaining
        ix1 = np.maximum(boxes[i, 0], boxes[order, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[order, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[order, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[order, 3])
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order, 2] - boxes[order, 0]) * (boxes[order, 3] - boxes[order, 1])
        iou = inter / np.maximum(area_i + area_j - inter, 1e-6)
        suppressed[order[iou >= iou_threshold]] = True
        suppressed[i] = False  # keep self

    return np.array(keep, dtype=np.intp)


def _linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[list, list, list]:
    """Solve linear assignment with IoU threshold.

    Returns (matches, unmatched_a, unmatched_b).
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_indices, col_indices = linear_sum_assignment(-cost_matrix)

    matched_a = set()
    matched_b = set()
    matches = []

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] >= thresh:
            matches.append((r, c))
            matched_a.add(r)
            matched_b.add(c)

    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_a]
    unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_b]

    return matches, unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# Helpers to vectorize track state access
# ---------------------------------------------------------------------------

def _get_boxes(tracks: list) -> np.ndarray:
    """Extract (N, 4) xyxy boxes from a list of STracks."""
    if not tracks:
        return np.empty((0, 4), dtype=np.float32)
    means = np.array([t.mean[:4] for t in tracks])  # (N, 4): cx, cy, w, h
    hw = means[:, 2:4] * 0.5  # half-width, half-height
    xy1 = means[:, :2] - hw
    xy2 = means[:, :2] + hw
    return np.hstack([xy1, xy2]).astype(np.float32)


def _stack_states(tracks: list) -> Tuple[np.ndarray, np.ndarray]:
    """Stack (N,8) means and (N,8,8) covariances from a track list."""
    means = np.array([t.mean for t in tracks])
    covs = np.array([t.covariance for t in tracks])
    return means, covs


def _write_states(tracks: list, means: np.ndarray, covs: np.ndarray) -> None:
    """Write batched states back into track objects."""
    for i, t in enumerate(tracks):
        t.mean = means[i]
        t.covariance = covs[i]


# ---------------------------------------------------------------------------
# BYTETracker
# ---------------------------------------------------------------------------

class BYTETracker:
    """ByteTrack multi-object tracker with vectorized Kalman operations.

    Three-stage association:
      1. Match high-score detections to active tracks via IoU (Hungarian)
      2. Match remaining low-score detections to unmatched active tracks
      3. Rescue: match remaining unmatched detections to lost tracks at
         lower IoU threshold (prevents ID switches)

    Optional built-in class-agnostic NMS preprocessing suppresses overlapping
    detections of different classes before association.

    Args:
        track_thresh: Score threshold to separate high/low detections.
        match_thresh: Minimum IoU for matching in first association.
        second_match_thresh: Minimum IoU for second association (low-score).
        lost_match_thresh: Minimum IoU for third association (lost rescue).
        max_time_lost: Remove tracks lost for this many frames.
        min_hits: Minimum consecutive hits before track is output.
        duplicate_iou_thresh: IoU above which two tracks are considered
            duplicates (the younger one is removed).
        class_agnostic_nms_thresh: If < 1.0, apply class-agnostic NMS to
            incoming detections before association. Set to 1.0 to disable.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.5,
        second_match_thresh: float = 0.4,
        lost_match_thresh: float = 0.3,
        max_time_lost: int = 30,
        min_hits: int = 3,
        duplicate_iou_thresh: float = 0.85,
        class_agnostic_nms_thresh: float = 1.0,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.lost_match_thresh = lost_match_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits
        self.duplicate_iou_thresh = duplicate_iou_thresh
        self.class_agnostic_nms_thresh = class_agnostic_nms_thresh

        self.kalman = KalmanFilter()
        self.frame_id = 0

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []

    def reset(self):
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        STrack.reset_id()

    def _remove_duplicate_tracks(self) -> None:
        """Remove duplicate tracked stracks (two tracks on the same object)."""
        n = len(self.tracked_stracks)
        if n < 2:
            return

        boxes = _get_boxes(self.tracked_stracks)
        iou = _iou_batch(boxes, boxes)

        remove = set()
        for i in range(n):
            if i in remove:
                continue
            for j in range(i + 1, n):
                if j in remove:
                    continue
                if iou[i, j] >= self.duplicate_iou_thresh:
                    ti = self.tracked_stracks[i]
                    tj = self.tracked_stracks[j]
                    if ti.tracklet_len >= tj.tracklet_len:
                        remove.add(j)
                    else:
                        remove.add(i)

        if remove:
            self.tracked_stracks = [
                t for idx, t in enumerate(self.tracked_stracks) if idx not in remove
            ]

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> List[STrack]:
        """Process one frame of detections.

        Args:
            boxes: (N, 4) detected boxes in xyxy format (original image coords).
            scores: (N,) detection scores.
            class_ids: (N,) integer class IDs.

        Returns:
            List of active STrack objects (each has .track_id, .box_xyxy,
            .score, .class_id).
        """
        self.frame_id += 1

        # ---- Optional class-agnostic NMS preprocessing ----
        if self.class_agnostic_nms_thresh < 1.0 and len(scores) > 0:
            keep = nms_class_agnostic(boxes, scores, class_ids, self.class_agnostic_nms_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

        # ---- Split into high / low score detections ----
        if len(scores) > 0:
            high_mask = scores >= self.track_thresh
            low_mask = ~high_mask & (scores >= 0.1)

            high_boxes = boxes[high_mask]
            high_scores = scores[high_mask]
            high_classes = class_ids[high_mask]

            low_boxes = boxes[low_mask]
            low_scores = scores[low_mask]
            low_classes = class_ids[low_mask]
        else:
            high_boxes = np.empty((0, 4), dtype=np.float32)
            high_scores = np.empty(0, dtype=np.float32)
            high_classes = np.empty(0, dtype=np.int64)
            low_boxes = np.empty((0, 4), dtype=np.float32)
            low_scores = np.empty(0, dtype=np.float32)
            low_classes = np.empty(0, dtype=np.int64)

        # ---- Batched Kalman predict for all existing tracks ----
        all_tracks = self.tracked_stracks + self.lost_stracks
        if all_tracks:
            means, covs = _stack_states(all_tracks)
            means, covs = self.kalman.predict_batch(means, covs)
            _write_states(all_tracks, means, covs)

        tracked_pool = list(self.tracked_stracks)
        lost_pool = list(self.lost_stracks)

        # ---- First association: high-score dets <-> tracked tracks ----
        if tracked_pool and len(high_boxes) > 0:
            track_boxes = _get_boxes(tracked_pool)
            iou_matrix = _iou_batch(track_boxes, high_boxes)
            matches, unmatched_tracks, unmatched_dets = _linear_assignment(
                iou_matrix, self.match_thresh
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(tracked_pool)))
            unmatched_dets = list(range(len(high_boxes)))

        # Batched Kalman update for matched tracks
        activated_stracks = []
        refound_stracks = []
        if matches:
            matched_tracks = [tracked_pool[t] for t, _ in matches]
            matched_dets_idx = [d for _, d in matches]
            det_boxes_matched = high_boxes[matched_dets_idx]  # (K, 4)

            # Convert xyxy detections to cxcywh measurements
            meas = np.column_stack([
                (det_boxes_matched[:, 0] + det_boxes_matched[:, 2]) * 0.5,
                (det_boxes_matched[:, 1] + det_boxes_matched[:, 3]) * 0.5,
                det_boxes_matched[:, 2] - det_boxes_matched[:, 0],
                det_boxes_matched[:, 3] - det_boxes_matched[:, 1],
            ])  # (K, 4)

            m_means, m_covs = _stack_states(matched_tracks)
            m_means, m_covs = self.kalman.update_batch(m_means, m_covs, meas)
            _write_states(matched_tracks, m_means, m_covs)

            for i, (t_idx, d_idx) in enumerate(matches):
                tracked_pool[t_idx].apply_update(
                    high_boxes[d_idx], float(high_scores[d_idx]),
                    int(high_classes[d_idx]), self.frame_id,
                )

        remaining_tracked = [tracked_pool[i] for i in unmatched_tracks]
        remaining_high_dets = list(unmatched_dets)

        # ---- Second association: low-score dets <-> remaining tracked ----
        if remaining_tracked and len(low_boxes) > 0:
            track_boxes = _get_boxes(remaining_tracked)
            iou_matrix = _iou_batch(track_boxes, low_boxes)
            matches2, unmatched_tracks2, _ = _linear_assignment(
                iou_matrix, self.second_match_thresh
            )
        else:
            matches2 = []
            unmatched_tracks2 = list(range(len(remaining_tracked)))

        # Batched Kalman update for second-round matches
        if matches2:
            matched_tracks2 = [remaining_tracked[t] for t, _ in matches2]
            det_idx2 = [d for _, d in matches2]
            det_boxes2 = low_boxes[det_idx2]
            meas2 = np.column_stack([
                (det_boxes2[:, 0] + det_boxes2[:, 2]) * 0.5,
                (det_boxes2[:, 1] + det_boxes2[:, 3]) * 0.5,
                det_boxes2[:, 2] - det_boxes2[:, 0],
                det_boxes2[:, 3] - det_boxes2[:, 1],
            ])
            m2_means, m2_covs = _stack_states(matched_tracks2)
            m2_means, m2_covs = self.kalman.update_batch(m2_means, m2_covs, meas2)
            _write_states(matched_tracks2, m2_means, m2_covs)

            for i, (t_idx, d_idx) in enumerate(matches2):
                remaining_tracked[t_idx].apply_update(
                    low_boxes[d_idx], float(low_scores[d_idx]),
                    int(low_classes[d_idx]), self.frame_id,
                )

        newly_lost = [remaining_tracked[i] for i in unmatched_tracks2]

        # ---- Third association (rescue): unmatched high dets <-> lost ----
        if lost_pool and remaining_high_dets:
            lost_boxes = _get_boxes(lost_pool)
            det_boxes_rem = high_boxes[remaining_high_dets]
            iou_matrix = _iou_batch(lost_boxes, det_boxes_rem)
            matches3, _, unmatched_dets3 = _linear_assignment(
                iou_matrix, self.lost_match_thresh
            )

            if matches3:
                matched_lost = [lost_pool[t] for t, _ in matches3]
                det_idx3 = [remaining_high_dets[d] for _, d in matches3]
                det_boxes3 = high_boxes[det_idx3]
                meas3 = np.column_stack([
                    (det_boxes3[:, 0] + det_boxes3[:, 2]) * 0.5,
                    (det_boxes3[:, 1] + det_boxes3[:, 3]) * 0.5,
                    det_boxes3[:, 2] - det_boxes3[:, 0],
                    det_boxes3[:, 3] - det_boxes3[:, 1],
                ])
                m3_means, m3_covs = _stack_states(matched_lost)
                m3_means, m3_covs = self.kalman.update_batch(m3_means, m3_covs, meas3)
                _write_states(matched_lost, m3_means, m3_covs)

                for i, (t_idx, d_idx) in enumerate(matches3):
                    real_d_idx = remaining_high_dets[d_idx]
                    lost_pool[t_idx].apply_update(
                        high_boxes[real_d_idx], float(high_scores[real_d_idx]),
                        int(high_classes[real_d_idx]), self.frame_id,
                    )
                    refound_stracks.append(lost_pool[t_idx])

            remaining_high_dets = [remaining_high_dets[i] for i in unmatched_dets3]

        # ---- Initialize new tracks ----
        for d_idx in remaining_high_dets:
            track = STrack(
                high_boxes[d_idx], float(high_scores[d_idx]),
                int(high_classes[d_idx]),
            )
            track.mean, track.covariance = self.kalman.initiate(track._measurement)
            track.activate(self.frame_id)
            activated_stracks.append(track)

        # ---- Update track lists ----
        refound_set = set(id(t) for t in refound_stracks)

        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.frame_id == self.frame_id
        ]
        self.tracked_stracks.extend(refound_stracks)
        self.tracked_stracks.extend(activated_stracks)

        self.lost_stracks = [
            t for t in self.lost_stracks
            if id(t) not in refound_set
            and (self.frame_id - t.frame_id) <= self.max_time_lost
        ]
        self.lost_stracks.extend(newly_lost)

        # ---- Deduplicate ----
        self._remove_duplicate_tracks()

        # ---- Output ----
        return [
            t for t in self.tracked_stracks
            if t.is_activated and t.tracklet_len >= self.min_hits
        ]
