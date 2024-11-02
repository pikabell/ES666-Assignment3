# Left to right stitching didn't worked for more than 3 images so used center to outward stitching along with cynlindrical
# warping to reduce distortion in the final panorama. Also the Field of view was quite big for some cases so used cylindrical
# Values of sensor width taken from "dpreview.com" for the respective camera models to get focal length in pixels
# SIFT gave better results than ORB so used SIFT for feature matching
# Used RANSAC to estimate best homography, the first homography from the best 4 points wasn't accurate enough
# compared the homography matrix with the one obtained from cv2.findHomography and found that the results were quite similar
# Used distance transform for blending the images, could've used laplacian pyramid for better blending but the
# results were satisfactory with distance transform
# Used ai to understand others open source code
# further improvements? running calculations on GPU, using laplacian pyramid for blending, using graph for finding the neighboring images (images need not be in left to right order in that case)

import pdb
import glob
import cv2
import os
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from PIL import Image

class PanaromaStitcher():
    """
    A class for stitching multiple images into a panoramic image.
    Uses cylindrical warping and SIFT features for better alignment.
    """

    def __init__(self):
        """Initialize the PanoramaStitcher class."""
        pass

    def get_focal_in_pixels(self, img_path):
        """
        Calculate the focal length in pixels from image EXIF data.

        Args:
            img_path (str): Path to the input image

        Returns:
            float: Focal length in pixels

        Note:
            Supports specific camera models: Canon DIGITAL IXUS 860 IS, DSC-W170,
            NEX-5N, and NIKON D40
        """

        sensor_width_map = {
            'Canon DIGITAL IXUS 860 IS': 5.75,
            'DSC-W170': 6.16,
            'NEX-5N': 23.4,
            'NIKON D40': 23.7
        }

        image = Image.open(img_path)
        focal_mm = image._getexif()[37386]  # Get focal length in mm from EXIF
        model = image._getexif()[272]       # Get camera model from EXIF
        sensor_width_mm = sensor_width_map[model]
        image_width_pixels = image.size[0]
        del image

        print(f"Focal length: {focal_mm}mm, Image width: {image_width_pixels}px, Sensor width: {sensor_width_mm}mm")
        focal_pixels = (focal_mm * image_width_pixels) / sensor_width_mm
        return focal_pixels

    def cylindricalWarp(self, img, f):
        """
        Apply cylindrical warping to an image to reduce distortion in panoramas.

        Args:
            img: Input image
            f: Focal length in pixels

        Returns:
            np.array: Warped image

        Note:
            This transformation helps reduce the distortion in the final panorama,
            especially for wide-angle scenes.
        """
        h, w = img.shape[:2]

        # Create intrinsic camera matrix based on focal length
        K = np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0, 1]])

        # Create pixel coordinate grid
        y_i, x_i = np.indices((h, w))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h * w, 3)

        # Convert to normalized coordinates
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T

        # Apply cylindrical projection
        A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w * h, 3)
        B = K.dot(A.T).T
        B = B[:, :-1] / B[:, [-1]]

        # Handle out-of-bounds points
        B[(B[:, 0] < 0) | (B[:, 0] >= w) | (B[:, 1] < 0) | (B[:, 1] >= h)] = -1
        B = B.reshape(h, w, -1)

        return cv2.remap(img, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32),
                        cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)

    def find_keypoint(self, image):
        """
        Detect SIFT keypoints and compute descriptors for an image.

        Args:
            image: Input image

        Returns:
            tuple: (keypoints, descriptors)
        """
        # Initialize SIFT detector with custom parameters for better detection
        sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)
        return sift.detectAndCompute(image, None)

    def match_keypoints(self, kp1, kp2, des1, des2):
        """
        Match keypoints between two images using ratio test.

        Args:
            kp1, kp2: Keypoints from first and second image
            des1, des2: Descriptors from first and second image

        Returns:
            tuple: (source points, destination points) for matched features
        """
        # Use Brute Force matcher with ratio test to find good matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test to filter good matches
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # Extract matching point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        return src_pts, dst_pts

    def homography(self, src_pts, dst_pts, max_iter=10000, inlier_thr=4):
        """
        Calculate homography matrix using RANSAC algorithm.

        Args:
            src_pts: Source points
            dst_pts: Destination points
            max_iter: Maximum RANSAC iterations
            inlier_thr: Threshold for considering inliers

        Returns:
            tuple: (best homography matrix, list of all computed matrices)
        """
        print(f"Computing homography with {len(src_pts)} point pairs")
        assert(len(src_pts) >= 4), "Need at least 4 point pairs"
        assert(len(dst_pts) == len(src_pts)), "Source and destination points must match"

        best_inlier = 0
        best_dist = float('inf')
        homography_matrices = []

        # RANSAC iteration
        for _ in range(max_iter):
            # Randomly select 4 point pairs
            idx = random.sample(range(len(src_pts)), 4)

            # Extract coordinates for selected points
            x1, x2, x3, x4 = ((src_pts[i][0], dst_pts[i][0]) for i in idx)
            y1, y2, y3, y4 = ((src_pts[i][1], dst_pts[i][1]) for i in idx)

            # Build the matrix for homography calculation
            P = np.array([
                [-x1[0], -y1[0], -1, 0, 0, 0, x1[0] * x1[1], y1[0] * x1[1], x1[1]],
                [0, 0, 0, -x1[0], -y1[0], -1, x1[0] * y1[1], y1[0] * y1[1], y1[1]],
                [-x2[0], -y2[0], -1, 0, 0, 0, x2[0] * x2[1], y2[0] * x2[1], x2[1]],
                [0, 0, 0, -x2[0], -y2[0], -1, x2[0] * y2[1], y2[0] * y2[1], y2[1]],
                [-x3[0], -y3[0], -1, 0, 0, 0, x3[0] * x3[1], y3[0] * x3[1], x3[1]],
                [0, 0, 0, -x3[0], -y3[0], -1, x3[0] * y3[1], y3[0] * y3[1], y3[1]],
                [-x4[0], -y4[0], -1, 0, 0, 0, x4[0] * x4[1], y4[0] * x4[1], x4[1]],
                [0, 0, 0, -x4[0], -y4[0], -1, x4[0] * y4[1], y4[0] * y4[1], y4[1]],
            ])

            # Compute homography using SVD
            [U, S, Vt] = np.linalg.svd(P)
            H = Vt[-1].reshape(3, 3)
            H /= H[2][2]  # Normalize homography matrix

            # Evaluate the quality of this homography
            pts = self.transform(src_pts, H)
            distvec = np.sqrt(np.sum(np.square(pts - dst_pts), axis=1))
            dist = np.mean(distvec[distvec < inlier_thr])
            inlier = np.count_nonzero(distvec < inlier_thr)

            homography_matrices.append(H)

            # Update best homography if current one is better
            if inlier > best_inlier or (inlier == best_inlier and dist < best_dist):
                best_inlier = inlier
                best_dist = dist
                best_H = H

        return best_H, homography_matrices

    def transform(self, src_pts, H):
        """
        Transform points using homography matrix.

        Args:
            src_pts: Source points
            H: Homography matrix

        Returns:
            np.array: Transformed points
        """
        # Convert to homogeneous coordinates
        src = np.pad(src_pts, [(0, 0), (0, 1)], constant_values=1)
        # Apply transformation
        pts = np.dot(H, src.T).T
        # Convert back to Euclidean coordinates
        pts = (pts / pts[:,-1].reshape(-1, 1))[:, 0:2]
        return pts

    def warp_perspective(self, img, H, shape):
        """
        Apply perspective warping to an image using a homography matrix.

        Args:
            img: Input image
            H: Homography matrix
            shape: Output image shape

        Returns:
            np.array: Warped image
        """
        h, w = shape
        y, x = np.indices((h, w))

        # Create homogeneous coordinates for all pixels
        dst_hom_pts = np.stack((x.ravel(), y.ravel(), np.ones(y.size)))

        # Apply inverse homography
        src_hom_pts = np.dot(np.linalg.inv(H), dst_hom_pts)
        src_hom_pts /= src_hom_pts[2]

        # Ensure coordinates are within image bounds
        src_x = np.clip(src_hom_pts[0].round().astype(int), 0, img.shape[1] - 1)
        src_y = np.clip(src_hom_pts[1].round().astype(int), 0, img.shape[0] - 1)

        # Create warped image
        warped_img = np.zeros((h, w, 3), dtype=img.dtype)
        warped_img[y.ravel(), x.ravel()] = img[src_y, src_x]

        return warped_img

    def stitch_images(self, img1, img2, homography):
        """
        Stitch two images together using homography and blending.

        Args:
            img1: First image
            img2: Second image
            homography: Homography matrix for transformation

        Returns:
            np.array: Stitched image
        """
        # Calculate corners of first image in homography space
        h, w = img1.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        transformed_corners = homography @ corners
        transformed_corners /= transformed_corners[2]

        # Calculate output dimensions
        min_x = min(0, transformed_corners[0].min())
        min_y = min(0, transformed_corners[1].min())
        max_x = max(img2.shape[1], transformed_corners[0].max())
        max_y = max(img2.shape[0], transformed_corners[1].max())

        # Create translation matrix
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        result_shape = (int(max_y - min_y), int(max_x - min_x), 3)
        result = np.zeros(result_shape, dtype=np.uint8)

        # Warp and blend images
        warp_matrix = translation @ homography
        warped_img1 = self.warp_perspective(img1, warp_matrix, result_shape[:2])
        result[-int(min_y):-int(min_y) + img2.shape[0],
               -int(min_x):-int(min_x) + img2.shape[1]] = img2

        # Create masks for blending
        mask_warped = (warped_img1.sum(axis=2) > 0).astype(np.uint8)
        mask_img2 = (result.sum(axis=2) > 0).astype(np.uint8)

        # Calculate blending weights using distance transform
        w1 = cv2.distanceTransform(mask_warped, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        w1 = cv2.normalize(w1, None, 0, 1.0, cv2.NORM_MINMAX)
        w2 = cv2.distanceTransform(mask_img2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        w2 = cv2.normalize(w2, None, 0, 1.0, cv2.NORM_MINMAX)

        # Compute final blending weights
        blend_w1 = w1 / (w1 + w2 + 1e-12)
        blend_w2 = w2 / (w1 + w2 + 1e-12)

        # Create final blended image
        final_image = (blend_w1[..., None] * warped_img1 +
                      blend_w2[..., None] * result).astype(np.uint8)
        return final_image

    def stitch_image(self, image1, image2):
        """
        Stitch two images together using feature matching and homography.

        Args:
            image1: First input image
            image2: Second input image

        Returns:
            tuple: (stitched image, homography matrix)
        """
        # Create copies to preserve original images
        img1 = image1.copy()
        img2 = image2.copy()

        # Find keypoints and descriptors
        kp1, des1 = self.find_keypoint(img1)
        kp2, des2 = self.find_keypoint(img2)

        # Match keypoints between images
        src_pts, dst_pts = self.match_keypoints(kp1, kp2, des1, des2)
        print(f"Found {len(src_pts)} matching points between images")

        # Calculate homography matrix
        H, homography_matrices = self.homography(src_pts, dst_pts)

        # Stitch images using calculated homography
        stitched = self.stitch_images(img1, img2, H)
        print("Stitching completed successfully")

        return stitched, H

    def make_panaroma_for_images_in(self, path):
        """
        Create a panorama from all images in a directory.

        Args:
            path: Directory containing input images

        Returns:
            tuple: (stitched panorama, list of homography matrices)
        """
        # Load and prepare images
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching')
        homography_matrix_list = []
        intermediate_image_list = []
        # Calculate focal lengths for all images
        focal_length = [self.get_focal_in_pixels(img) for img in all_images]

        # Load and warp all images
        all_images = [cv2.imread(img) for img in all_images]
        print('Focal Lengths:', focal_length)

        # Apply cylindrical warping to reduce distortion
        warped_images = [self.cylindricalWarp(img, f) for img, f in zip(all_images, focal_length)]
        all_images = warped_images

        # Find middle image to start stitching from
        mid = 0
        stitched_image = np.zeros((1, 1, 3), dtype=np.uint8)

        # Handle even number of images
        if len(all_images) % 2 == 0:  # 0 1 {2 3} 4 5
            mid = len(all_images) // 2
            left_images = all_images[:mid-1]
            right_images = all_images[mid+1:]
            # Start with middle two images
            stitched_image, H = self.stitch_image(all_images[mid-1], all_images[mid])
        else:  # 0 1 {2} 3 4
            # For odd number of images, start with middle image
            mid = len(all_images) // 2
            left_images = all_images[:mid]
            right_images = all_images[mid+1:]
            stitched_image = all_images[mid]

        print(f"Starting with {len(left_images)} left images and {len(right_images)} right images")

        # Process images from center outward
        left_images = left_images[::-1]  # Reverse left images for proper order

        # Stitch remaining images
        for i in range(len(left_images)):
            # Stitch right image
            stitched_image, homo = self.stitch_image(right_images[i], stitched_image)
            homography_matrix_list.append(homo)
            intermediate_image_list.append(stitched_image)
            # Stitch left image
            stitched_image, homo = self.stitch_image(left_images[i], stitched_image)
            homography_matrix_list.append(homo)
            intermediate_image_list.append(stitched_image)

        return stitched_image, homography_matrix_list , intermediate_image_list