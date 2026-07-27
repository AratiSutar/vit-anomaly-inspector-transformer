import numpy as np
import pickle
import os


def find_optimal_threshold(scores, labels, save_path='outputs', category='bottle'):
    scores = np.array(scores)
    labels = np.array(labels)

    # try every unique score as a possible threshold
    thresholds = np.unique(scores)

    best_threshold = 0
    best_f1 = 0

    for t in thresholds:
        predictions = (scores > t).astype(int)

        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"Optimal threshold: {best_threshold:.4f} | Best F1: {best_f1:.4f}")

    os.makedirs(save_path, exist_ok=True)
    threshold_file = os.path.join(save_path, f'{category}_threshold.pkl')
    with open(threshold_file, 'wb') as f:
        pickle.dump({'threshold': best_threshold, 'f1': best_f1}, f)

    print(f"Threshold saved to {threshold_file}")
    return best_threshold, best_f1
