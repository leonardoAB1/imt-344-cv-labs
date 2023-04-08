# Helper functions

import os
import glob # library for loading images from a directory
import matplotlib.image as mpimg




# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["red", "yellow", "green"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list


def region_growing(img, seed, tolerance=0.1, color=255):
    """
    Segment an image using region growing.
    
    Args:
        img (np.ndarray): The input image.
        seed (tuple): The pixel coordinates (x, y) of the seed point.
        tolerance (float): Allowed difference between seed value and neighborhood
        color (int) = mask color
    
    Returns:
        np.ndarray: The segmented image mask.
    """
    # Initialize segmented image with zeros
    segmented = np.zeros_like(img)
    
    # Initialize queue with seed point
    queue = [seed]
    
    # Get seed point color
    seed_color = img[seed]
    
    # Loop until the queue is empty
    while queue:
        # Get next pixel from queue
        pixel = queue.pop(0)
        
        # Check if pixel is within image bounds
        if (0 <= pixel[0] < img.shape[0]) and (0 <= pixel[1] < img.shape[1]):
            # Check if pixel is unsegmented and similar to seed color
            if segmented[pixel[0], pixel[1]] == 0 and np.allclose(img[pixel], seed_color, rtol=tolerance, atol=tolerance):
                # Segment pixel
                segmented[pixel[0], pixel[1]] = color
                
                # Add neighboring pixels to queue
                queue.append((pixel[0] - 1, pixel[1]))
                queue.append((pixel[0] + 1, pixel[1]))
                queue.append((pixel[0], pixel[1] - 1))
                queue.append((pixel[0], pixel[1] + 1))
    
    return segmented

