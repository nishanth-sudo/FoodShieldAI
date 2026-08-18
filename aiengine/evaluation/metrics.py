import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ModelMetrics:
    @staticmethod
    def classification_metrics(
        y_true: np.ndarray | torch.Tensor,
        y_pred: np.ndarray | torch.Tensor,
    ) -> dict:
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            y_pred_classes = y_pred.argmax(axis=1)
        else:
            y_pred_classes = (
                (y_pred > 0.5).astype(np.int32)
                if y_pred.ndim == 2
                else y_pred.round().astype(np.int32)
            )

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true_classes = y_true.argmax(axis=1)
        else:
            y_true_classes = y_true.astype(np.int32).squeeze()

        correct = (y_pred_classes == y_true_classes).sum()
        total = len(y_true_classes)
        accuracy = float(correct) / float(total) if total > 0 else 0.0

        n_classes = int(max(y_true_classes.max(), y_pred_classes.max())) + 1
        conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(y_true_classes, y_pred_classes, strict=True):
            conf_matrix[t, p] += 1

        precision_list = []
        recall_list = []
        f1_list = []
        for c in range(n_classes):
            tp = conf_matrix[c, c]
            fp = conf_matrix[:, c].sum() - tp
            fn = conf_matrix[c, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        macro_precision = float(np.mean(precision_list))
        macro_recall = float(np.mean(recall_list))
        macro_f1 = float(np.mean(f1_list))

        if n_classes == 2:
            tp = conf_matrix[1, 1]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]
            tn = conf_matrix[0, 0]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        return {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "specificity": round(specificity, 4),
            "confusion_matrix": conf_matrix.tolist(),
            "per_class_precision": [round(p, 4) for p in precision_list],
            "per_class_recall": [round(r, 4) for r in recall_list],
            "per_class_f1": [round(f, 4) for f in f1_list],
            "n_classes": int(n_classes),
            "n_samples": int(total),
        }

    @staticmethod
    def detection_metrics(
        y_true: list[dict],
        y_pred: list[dict],
        iou_threshold: float = 0.5,
    ) -> dict:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        iou_scores: list[float] = []

        for gt, pred in zip(y_true, y_pred, strict=True):
            gt_boxes = gt.get("boxes", [])
            pred_boxes = pred.get("boxes", [])

            matched_gt = set()
            for p_box in pred_boxes:
                best_iou = 0.0
                best_gt_idx = -1
                for i, g_box in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    iou = ModelMetrics._compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    iou_scores.append(best_iou)
                else:
                    false_positives += 1

            false_negatives += len(gt_boxes) - len(matched_gt)

        total_tp_fp = true_positives + false_positives
        precision = true_positives / total_tp_fp if total_tp_fp > 0 else 0.0
        total_tp_fn = true_positives + false_negatives
        recall = true_positives / total_tp_fn if total_tp_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
            "average_precision": round(precision, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "iou_threshold": iou_threshold,
        }

    @staticmethod
    def regression_metrics(
        y_true: np.ndarray | torch.Tensor,
        y_pred: np.ndarray | torch.Tensor,
    ) -> dict:
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        y_true = y_true.flatten().astype(np.float64)
        y_pred = y_pred.flatten().astype(np.float64)

        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        squared_errors = errors**2

        mae = float(np.mean(abs_errors))
        mse = float(np.mean(squared_errors))
        rmse = float(np.sqrt(mse))
        mape_val = float(np.mean(np.abs(errors / (y_true + 1e-8)))) * 100.0

        ss_res = float(np.sum(squared_errors))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape_val, 4),
            "r2_score": round(r2, 4),
            "max_error": round(float(np.max(abs_errors)), 4),
            "n_samples": int(len(y_true)),
        }

    @staticmethod
    def _compute_iou(box1: list[float], box2: list[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
