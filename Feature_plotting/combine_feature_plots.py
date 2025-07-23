import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_common_feature_names(folders, suffix):
    """Find plot filenames with the given suffix that are shared across all folders."""
    file_sets = [set(f for f in os.listdir(folder) if f.endswith(suffix)) for folder in folders]
    return sorted(set.intersection(*file_sets))

def combine_plots(folders, labels, output_dir, suffix, subfolder):
    os.makedirs(output_dir, exist_ok=True)
    output_subdir = os.path.join(output_dir, subfolder)
    os.makedirs(output_subdir, exist_ok=True)

    common_files = get_common_feature_names(folders, suffix)

    for feature_file in common_files:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10)) if len(folders) == 4 else plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
        for ax, folder, label in zip(axes, folders, labels):
            img_path = os.path.join(folder, feature_file)
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(label)
            ax.axis("off")

        output_path = os.path.join(output_subdir, feature_file)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[INFO] Combined plot saved to {output_path}", flush=True)

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Train directory '{args.train_dir}' does not exist.")
    if not os.path.exists(args.val_dir):
        raise FileNotFoundError(f"Validation directory '{args.val_dir}' does not exist.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Test directory '{args.test_dir}' does not exist.")
    
    
    if not os.path.exists(args.gutenberg_dir):
        print(f"[WARNING] Gutenberg directory '{args.gutenberg_dir}' does not exist. Skipping.")
        folders = [args.train_dir, args.val_dir, args.test_dir]
        labels = ["train", "val", "test"]
    else:
        print(f"[INFO] Using Gutenberg directory '{args.gutenberg_dir}' for additional plots.")
        folders = [args.train_dir, args.val_dir, args.test_dir, args.gutenberg_dir]
        labels = ["train", "val", "test", "gutenberg"]

    combine_plots(folders, labels, args.output_dir, "_mean_by_{}.png".format(args.target), "mean")
    combine_plots(folders, labels, args.output_dir, "_box_by_{}.png".format(args.target), "box")
    combine_plots(folders, labels, args.output_dir, "_violin_by_{}.png".format(args.target), "violin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine train/val/test/gutenberg feature plots into 2x2 grids")
    parser.add_argument("--train_dir", required=True, help="Directory with train plots")
    parser.add_argument("--val_dir", required=True, help="Directory with validation plots")
    parser.add_argument("--test_dir", required=True, help="Directory with test plots")
    parser.add_argument("--gutenberg_dir", required=True, help="Directory with gutenberg plots")
    parser.add_argument("--output_dir", default="combined_feature_plots", help="Where to save combined plots")
    parser.add_argument("--target", default="decade", help="Grouping column used in plots (e.g., 'decade' or 'century')")
    args = parser.parse_args()
    main(args)
