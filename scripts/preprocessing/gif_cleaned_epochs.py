import mne
from PIL import Image
import os
import tempfile
import shutil
import matplotlib.pyplot as plt
import argparse
from joblib import Parallel, delayed

def generate_gif_from_epochs_in_groups(epochs, output_path, epochs_per_image=5, duration=500):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    images = []
    
    # Loop through epochs in groups of `epochs_per_image`
    total_epochs = len(epochs)
    num_images = (total_epochs + epochs_per_image - 1) // epochs_per_image  # Ceiling division

    for img_idx in range(num_images):
        start_epoch = img_idx * epochs_per_image
        end_epoch = min(start_epoch + epochs_per_image, total_epochs)
        print(f"Processing epochs {start_epoch} to {end_epoch-1}")
        # Plot and save the group of epochs as a single image
        fig = epochs[start_epoch:end_epoch].plot(show=False, n_channels=271)
        img_path = os.path.join(temp_dir, f'epochs_{start_epoch}-{end_epoch-1}.png')
        fig.savefig(img_path, dpi=500)
        images.append(img_path)
        plt.close(fig)  # Close the figure to free memory

    # Load all saved images and create a single GIF
    frames = [Image.open(img) for img in images]
    frames[0].save(output_path, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration,
                   loop=0)

    # Cleanup temporary directory
    shutil.rmtree(temp_dir)

    print(f"Generated GIF for all epochs in groups: {output_path}")

def process_subject(subject_id, task, condition, epochs_per_image):
    # Construct the file path for the cleaned epochs
    output_dir = f"/users/local/Venkatesh/LSD_project/src_data/derivatives/func/{task}/{condition}/sub-{subject_id}/meg/"
    
    epochs_file = f"/users/local/Venkatesh/LSD_project/src_data/derivatives/func/{task}/{condition}/sub-{subject_id}/meg/sub-{subject_id}_cleaned_epochs_meg.fif"
    epochs = mne.read_epochs(epochs_file)
    duration = len(epochs)*2
    output_path = os.path.join(output_dir, f'{output_dir}_sub-{subject_id}_cleaned_epochs.gif')
    generate_gif_from_epochs_in_groups(epochs, output_path, epochs_per_image, duration)

def main():
    parser = argparse.ArgumentParser(description="Generate GIFs for EEG epochs.")
    parser.add_argument('--subjects', type=str, nargs='+', required=True, help="List of subject IDs.")
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help="List of tasks (e.g., Music, Video).")
    parser.add_argument('--conditions', type=str, nargs='+', required=True, help="List of conditions (e.g., LSD, PLA).")
    parser.add_argument('--epochs_per_image', type=int, default=20, help="Number of epochs per image in the GIF.")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of jobs for parallel processing.")
    
    args = parser.parse_args()

    
    # Process each combination of subject, task, and condition in parallel
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_subject)(subject_id, task, condition, args.epochs_per_image)
        for subject_id in args.subjects
        for task in args.tasks
        for condition in args.conditions
    )

if __name__ == "__main__":
    main()
