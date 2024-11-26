import os
import shutil

def reorganize_data(data_dir):
    # Create channel directories
    for channel in range(3):
        channel_dir = os.path.join(data_dir, f'channel{channel}')
        os.makedirs(channel_dir, exist_ok=True)

    # Iterate over each image directory
    for image_index in range(1, 121):
        image_dir = os.path.join(data_dir, f'Image{str(image_index).zfill(3)}')
        if not os.path.exists(image_dir):
            continue

        # Move files to the corresponding channel directories
        for channel in range(3):
            channel_dir = os.path.join(data_dir, f'channel{channel}')
            new_image_dir = os.path.join(channel_dir, f'Image{str(image_index).zfill(3)}')
            os.makedirs(new_image_dir, exist_ok=True)

            # Move wf_channel files
            wf_file = os.path.join(image_dir, f'wf_channel{channel}.npy')
            if os.path.exists(wf_file):
                shutil.move(wf_file, os.path.join(new_image_dir, f'wf_channel{channel}.npy'))

            # Delete sim_channel files
            sim_file = os.path.join(image_dir, f'sim_channel{channel}.npy')
            if os.path.exists(sim_file):
                os.remove(sim_file)

            # Delete sim_input_channel files
            sim_input_file = os.path.join(image_dir, f'sim_input_channel{channel}.npy')
            if os.path.exists(sim_input_file):
                os.remove(sim_input_file)

        # Remove the original image directory if empty
        if not os.listdir(image_dir):
            os.rmdir(image_dir)
            


def reorganize_data_back(data_dir):
    # Iterate over each channel directory
    for channel in range(3):
        channel_dir = os.path.join(data_dir, f'channel{channel}')
        if not os.path.exists(channel_dir):
            continue

        # Iterate over each image directory within the channel directory
        for image_index in range(1, 121):
            new_image_dir = os.path.join(channel_dir, f'Image{str(image_index).zfill(3)}')
            if not os.path.exists(new_image_dir):
                continue

            # Create the original image directory if it doesn't exist
            original_image_dir = os.path.join(data_dir, f'Image{str(image_index).zfill(3)}')
            os.makedirs(original_image_dir, exist_ok=True)

            # Move wf_channel files back to the original image directory
            wf_file = os.path.join(new_image_dir, f'wf_channel{channel}.npy')
            if os.path.exists(wf_file):
                shutil.move(wf_file, os.path.join(original_image_dir, f'wf_channel{channel}.npy'))

            # Remove the new image directory if empty
            if not os.listdir(new_image_dir):
                os.rmdir(new_image_dir)

        # Remove the channel directory if empty
        if not os.listdir(channel_dir):
            os.rmdir(channel_dir)


if __name__ == "__main__":
    data_dir = 'data'  # Path to the data directory
    reorganize_data(data_dir)
    # reorganize_data_back(data_dir)
