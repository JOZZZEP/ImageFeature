import cv2
# Load the image as grayscale
# img_gray = cv2.imread('135.jpg', 0)

def gethog(img):
    new_img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
    win_size = new_img.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    # Set the parameters of the HOG descriptor using the variables defined above
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,cell_size, num_bins)
    # Compute the HOG Descriptor for the gray scale image
    hog_descriptor = hog.compute(new_img)
    return hog_descriptor
    
# print(gethog(img_gray))