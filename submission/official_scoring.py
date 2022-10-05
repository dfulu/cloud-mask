from pathlib import Path
import numpy as np
from tifffile import imread
from tqdm import tqdm


def iterate_through_mask_pairs(submission_dir: Path, actual_dir: Path):
    """
    For each tif in the actual directory, find the corresponding prediction tif, read
    them both in, and yield the (pred, actual) tuple
    """
    for predicted_path in submission_dir.glob("*.tif"):
        filename = predicted_path.name
        actual_path = actual_dir / filename
        assert actual_path.exists(), f"Could not find expected file: {filename}"
        actual = imread(actual_path)
        pred = imread(predicted_path)
        yield pred, actual


def intersection_over_union(array_pairs, total=None):
    """Calculate the actual metric"""
    intersection = 0
    union = 0
    for pred, actual in tqdm(array_pairs, total=total):
        intersection += np.logical_and(actual, pred).sum()
        union += np.logical_or(actual, pred).sum()
    if union < 1:
        raise ValueError("At least one image must be in the actual data set")
    return intersection / union


def main(submission_dir: Path, actual_dir: Path):
    """
    Given a directory with the predicted mask files (all values in {0, 1}) and the actual
    mask files (all values in {0, 1}), get the overall intersection-over-union score
    """
    n_expected = len(list(submission_dir.glob("*.tif")))
    array_pairs = iterate_through_mask_pairs(submission_dir, actual_dir)
    print(f"calculating score for {n_expected} image pairs ...")
    score = intersection_over_union(array_pairs, total=n_expected)
    print(f"overall score: {score}")
    return score

submission_dir = Path("/home/s1205782/Datastore/Projects/cloudmask/data/submission_test/predictions")
actual_dir = Path("/home/s1205782/Datastore/Projects/cloudmask/data/train_labels")
main(submission_dir, actual_dir)