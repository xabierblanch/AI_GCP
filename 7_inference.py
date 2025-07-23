# Import required libraries
import cv2
import glob
import math
import numpy as np
import os
import shutil
from pathlib import Path

import torch
import torchvision
from PIL import Image
from sklearn.cluster import KMeans
# Dataset and DataLoader classes from PyTorch's torch.utils.data module
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms


def split_images(img_dir, patch_size, overlap):
    """
    Split all images in a folder into smaller patches with a specified size and overlap.

    Parameters:
        img_dir (str): The directory containing the input images.
        patch_size (int): The size (in pixels) of the patches to split the images into.
        overlap (float): The fractional overlap between patches.

    Returns:
        str: The path to the directory containing the cropped patches.
    """
    # Get a list of all files in the folder
    file_list = os.listdir(img_dir)

    # Create a directory to store the cropped patches
    os.makedirs(f'{img_dir}/_crop', exist_ok=True)

    # Iterate through the list of files
    for file_name in tqdm(file_list, desc='Splitting images'):
        # Check if the file is an image
        _, file_ext = os.path.splitext(file_name)
        if file_ext in ['.jpg', '.png', '.jpeg', '.bmp', '.JPG']:
            # Open the image
            im = Image.open(os.path.join(img_dir, file_name))

            # Calculate the overlap size in pixels
            overlap_size = int(patch_size * overlap)

            # Calculate the number of rows and columns of patches taking into account the overlap
            rows = math.ceil(im.height / (patch_size - overlap_size))
            cols = math.ceil(im.width / (patch_size - overlap_size))

            # Iterate through the patches and save them
            for i in range(rows):
                for j in range(cols):
                    # Create a blank image with the appropriate size
                    patched_image = Image.new("RGB", (patch_size, patch_size))
                    # Calculate the bounds of the patch
                    left = j * patch_size - overlap_size * j
                    top = i * patch_size - overlap_size * i
                    right = min(left + patch_size, im.width)
                    bottom = min(top + patch_size, im.height)

                    # Crop the patch from the image
                    patch = im.crop((left, top, right, bottom))
                    # Paste the patch into the patched image
                    patched_image.paste(patch, (0, 0))

                    # Save the patch with an encoded name that includes the patch size, overlap, number of rows and columns, and original file name
                    patched_image.save(f'{img_dir}/_crop/{Path(file_name).stem}_patch_{patch_size}_{overlap}_{rows}_{cols}_{i}_{j}.jpg')

    return os.path.join(img_dir,'_crop')


class ImageDataset(Dataset):
    def __init__(self, crop_dir, mean, std):
        # Store the directory where the image tiles are located
        self.crop_dir = crop_dir
        # Get a list of all image tiles in the directory
        self.images = os.listdir(self.crop_dir)
        print(f'Dataset initialized with {len(self.images)} images')

        if mean != None and std != None:
            print(f'Dataset normalized using mean and std\n')
            self.std = torch.tensor(mean)
            self.mean = torch.tensor(std)
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.mean,
                                                                      std=self.std)])
        else:
            print(f'Dataset not normalized\n')
            self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        # Get the path to the image tile
        image_path = os.path.join(self.crop_dir, self.images[idx])
        # Open the image
        image = Image.open(image_path)
        # Convert the image to a tensor
        image = torchvision.transforms.functional.to_tensor(image)
        return image, self.images[idx]

    def __len__(self):
        # Return the number of image tiles in the dataset
        return len(self.images)


def get_model(model_path):
    # Check if a GPU is available, else use the CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # Load the model from the specified path
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model = checkpoint['model']

    # Load the model weights from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    # Move the model to the selected device
    model.to(device)

    return model, device


def inference(data_loader, model, device, treshold):
    """
    Perform inference on each element in the given data loader.

    Parameters:
    data_loader (DataLoader): Data loader containing the data to infer on.
    model (nn.Module): Model to use for inference.
    device (torch.device): Device to run inference on.

    Returns:
    tuple: Tuple containing the paths to the output keypoint and image directories.
    """

    # Get parent directory of crop directory
    root = Path(data_loader.dataset.crop_dir).parent
    # Set output keypoint and image directories
    output_keypoint = os.path.join(root, '_inference')
    # Create output directories if they don't exist
    os.makedirs(output_keypoint, exist_ok=True)

    # Iterate over each element in the inference data set
    for test_features, name in tqdm(data_loader, desc='Inferencing images'):
        # Convert test features to specified device
        input = test_features.to(device=device)
        # Make predictions using provided model
        predictions = model(input)

        # Iterate over each input image
        for y, _ in enumerate(input):
            # Get bounding boxes, keypoints and scores from predictions
            bboxes = predictions[y]['boxes'].detach().cpu().numpy()
            keypoints = predictions[y]['keypoints'].detach().cpu().numpy()
            scores = predictions[y]['scores'].detach().cpu().numpy()

            # Filter bounding boxes with score greater than or equal to 0.95
            idx = scores > treshold

            # Apply non-maximum suppression to bounding boxes and keypoints to remove redundancies
            post_nms_idxs = torchvision.ops.nms(predictions[y]['boxes'][idx], predictions[y]['scores'][idx],
                                                0).cpu().numpy()

            # Initialize empty list for keypoints
            file = []
            # Iterate over keypoints and add them to list
            for i, (keypoint) in enumerate(keypoints[post_nms_idxs]):
                file.append([keypoint[0][0],keypoint[0][1],
                             scores[post_nms_idxs][i],
                             bboxes[post_nms_idxs][i][0],bboxes[post_nms_idxs][i][1],
                             bboxes[post_nms_idxs][i][2],bboxes[post_nms_idxs][i][3]])
            # Convert keypoints list to numpy array
            file = np.array(file)
            # Save keypoints array to file in output keypoint directory
            np.save(os.path.join(output_keypoint, Path(name[y]).stem), file)

    return output_keypoint

def merge_npy(output_keypoint):
    """
    Merge a set of patches by combining their keypoint coordinates into a single file for each image.

    Parameters:
    output_keypoint (str): The path to the directory containing the keypoint files for the patches.

    Returns:
    str: The path to the directory containing the merged keypoint files.
    """
    # Create a directory to store the merged keypoint files
    output_path = os.path.join(os.path.dirname(output_keypoint), 'result/gcp')
    os.makedirs(output_path, exist_ok=True)

    # Get a list of all keypoint files in the output_keypoint directory
    file_list = os.listdir(output_keypoint)
    files = []

    # Iterate through the list of files and extract the base file names
    for patch_file in file_list:
        files.append(patch_file.split('_patch')[0])

    # Remove duplicates from the list of image names
    files = set(files)

    for xyz_file in tqdm(files, desc='Merging coordinates'):
        # Get a list of all keypoint files for the current image
        npy_list = glob.glob(output_keypoint + f'/{xyz_file}*')
        keypoints = []

        # Iterate through the patches and paste them onto the final image
        for npy_file in npy_list:
            # Read the patch size, overlap, number of rows and columns from the patch file names
            npy_code = Path(npy_file).name.split('_patch_')[-1].split('_')
            img_name = Path(npy_file).name.split('_patch_')[0]
            patch_size = int(npy_code[0])
            overlap = float(npy_code[1])
            j = int(npy_code[4])
            i = int(npy_code[5].split('.')[0])

            # Calculate the size of the final image
            overlap_size = int(patch_size * overlap)
            xyz = np.load(npy_file)

            # Skip the patch if it doesn't contain any keypoints
            if len(xyz) == 0:
                continue

            if xyz[0][3] > 0 and xyz[0][4] > 0 and xyz[0][5] < patch_size and xyz[0][6] < patch_size:
                # Calculate the position of the patch on the final image
                left = j * patch_size - overlap_size * j
                top = i * patch_size - overlap_size * i
                y_coord = xyz[0][1] + left
                x_coord = xyz[0][0] + top
                bbox_x = xyz[0][3] + top
                bbox_y = xyz[0][4] + left
                bbox_xx = xyz[0][5] + top
                bbox_yy = xyz[0][6] + left
                keypoints.append([x_coord,y_coord,xyz[0][2],bbox_x, bbox_y, bbox_xx, bbox_yy])
            else:
                continue
        np.savetxt(f'{output_path}/{img_name}.txt', np.array(keypoints), fmt='%.5f')
    return output_path


def create_images(img_dir, gcp_path):
    """
    Creates images with keypoints overlaid on the original image.

    Parameters:
        img_dir (str): The directory containing the input images.
        gcp_path (str): The directory containing the keypoints for the input images.

    Returns:
        None
    """
    # Create a directory for the result images if it does not already exist
    os.makedirs(f'{img_dir}/result/img', exist_ok=True)

    # Iterate through all files in the gcp_path directory
    for file in tqdm(os.listdir(gcp_path), desc='Creating images'):
        # Load the keypoints from the file
        xyz = np.loadtxt(f'{gcp_path}/{file}')
        # Load the image
        img = cv2.imread(f'{img_dir}/{Path(file).stem}.JPG')
        # Convert the image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Iterate through all keypoints in the file
        for line in xyz:
            # Draw a circle at the keypoint location
            img = cv2.circle(img.copy(), (int(line[0]), int(line[1])), radius=0, color=[255, 0, 0], thickness=3)
            # Draw a rectangle around the keypoint
            img = cv2.rectangle(img.copy(), pt1=(int(line[3]), int(line[4])), pt2=(int(line[5]), int(line[6])),
                                color=(0, 0, 255), thickness=2)
        # Save the image with the keypoints overlaid
        cv2.imwrite(f'{img_dir}/result/img/{file}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def removing_temp_folders(keypoint_path, crop_path):
    """
    This function removes the specified directories and all their contents.

    Parameters:
    keypoint_path (str): The path of the keypoint folder to be deleted.
    crop_path (str): The path of the crop folder to be deleted.

    Returns:
    None
    """
    try:
        # delete the folder and all its contents
        shutil.rmtree(keypoint_path)
        shutil.rmtree(crop_path)

    except OSError as e:
        # handle errors
        print("Error: %s" % (e.strerror))


def refine_coordinates(gcp_path):
    """
    This function clusters keypoints with similar coordinates and replaces them with a single keypoint at the weighted average
    position, with the score being the average of the scores of the keypoints in the cluster. It also refines the bounding box
    coordinates using the same method.

    Parameters:
    gcp_path (str): The path of the directory containing the keypoint files to be refined.

    Returns:
    None
    """
    # Iterate over each file in the gcp_path directory
    for file in tqdm(os.listdir(gcp_path), desc='Refining coordinates'):
        # Load the coordinates from the file
        coordinates = np.loadtxt(f'{gcp_path}/{file}')

        # Split the coordinates into keypoints and accuracies
        keypoint = coordinates[:, 0:2]
        accuracies = coordinates[:, 2]

        # Initialize the number of clusters and the weighted average distance threshold
        n_clusters = 0
        threshold = 1
        avg_distance = 100

        # While the weighted average distance is greater than the threshold, increase the number of clusters
        while avg_distance > threshold:
            n_clusters += 1
            # Fit a KMeans model to the keypoints using the accuracies as weights
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(keypoint, sample_weight=accuracies)
            labels = kmeans.labels_
            distances = kmeans.transform(keypoint).min(axis=1)
            avg_distance = np.average(distances, weights=accuracies)

        # Create a list to store the refined coordinates for each cluster
        refined_coordinates = []

        # Iterate over each cluster label
        for label in range(n_clusters):
            # Select the coordinates for the current cluster
            cluster_coordinates = coordinates[labels == label]

            # Calculate the weighted averages for all columns
            point_avgs = np.average(cluster_coordinates, axis=0, weights=cluster_coordinates[:, 2])

            # Add the coordinates for the current cluster to the list
            refined_coordinates.append(point_avgs)

        np.savetxt(f'{gcp_path}/{file}', np.array(refined_coordinates), fmt='%.5f')


if __name__ == '__main__':
    # Set variables for input image directory, model path, patch size, overlap, and threshold
    img_dir = '/home/xbg/Desktop/Test_images/raw/cam1/img'
    model_path = '/mnt/SSD_Data/Xabier/GITLAB_(XBG)/GCP_Detection/Model_Zoo/Pulmuki/20230807_1043_Keypoint/2023-08-08_e369_n88.17__DstPx-0.405_DArea-1.33_Score-0.993_rcnn-resnet50.pth'
    patch_size = 512 #best performance -> same than training
    overlap = 0.3 #overlap in % over the patch_size
    threshold = 0.5 #threshold of the inference step
    batch_size = 10 #batch size for the inference step

    mean = [0.44271782, 0.437117, 0.36180425]
    std = [0.14533198, 0.14259796, 0.14833033]

    # Split the image into patches
    crop_path = split_images(img_dir, patch_size, overlap)
    # crop_path = img_dir

    # Load the model
    model, device = get_model(model_path)

    # Create a dataset for the image patches
    dataset_inference = ImageDataset(crop_path, mean, std)

    # Create a data loader for the dataset
    data_loader = DataLoader(dataset_inference, batch_size=batch_size, shuffle=False)

    # Run inference on the image patches
    keypoint_path = inference(data_loader, model, device, threshold)

    # Merge the npy files containing the keypoints
    gcp_path = merge_npy(keypoint_path)

    # Refine the coordinates using K-means clustering
    refine_coordinates(gcp_path)

    # Create images of the keypoints overlaid on the original image
    create_images(img_dir, gcp_path)

    # Remove the temporary folders and files
    removing_temp_folders(keypoint_path, crop_path)