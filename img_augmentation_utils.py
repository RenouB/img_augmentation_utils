import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def get_crop_indices_from_mask(mask):
    """
    uses a mask to get cropping indices, so that images can be cropped to minimum
    rectangle covering mask

    param mask: binary mask with masked pixels == 255
    return: min, max x and y coords
    """
    ys, xs = np.where(mask == 255)
    return min(ys), max(ys), min(xs), max(xs)


def get_polygon_centre(polygon):
    """
    get centre coords of polygon
    params polygon: list / array of vertex coords

    return: array [centre_y, centre_x]
    """
    if type(polygon) != np.ndarray:
        polygon = np.array(polygon)
    polygon = np.squeeze(polygon, axis=1)
    min_y = np.min(polygon[:,0])
    min_x = np.min(polygon[:,1])
    max_y = np.max(polygon[:,0])
    max_x = np.max(polygon[:,1])
    centre_x = max_x - (max_x - min_x) / 2
    centre_y = max_y - (max_y - min_y) / 2

    return np.array([centre_y, centre_x])


def enlargen_polygon(polygon, ratio):
    """
    takes a polygon and expands it by increasing length of vectors from centre to
    all corners

    params polygon: list / array of vertex coords
    ratio: ratio by which to lengthen vectors from centre to vertices

    return: enlargened_polygon
    """
    centre = get_polygon_centre(polygon)
    polygon = polygon.astype(np.int)

    enlargened_poly = []
    for corner in polygon:
        diff = corner - centre
        enlargened_poly.append((diff * ratio) + centre)
    return np.array(enlargened_poly).astype(np.int32)


def get_padded_polygon_image(enlargened_poly, img, mask, color=255):
    """
    uses enlargened polygon to add a white border around masked portion
    of original image

    param enlargened_poly: list / array of vertices describing polygon
          img: 3d opencv image
          mask: 2d binary mask with masked pixels == 255
          color: color of padding in padded image

    return: padded image, color image of original with white padding around
            polygon
            padded_mask, a binary mask that indicates area of original polygon
            + added border
    """

    # mask to extract area of interest
    extracted_img = cv2.bitwise_and(img, img, mask=mask)
    # invert mask
    mask_inv = cv2.bitwise_not(mask)

    padded_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(padded_mask, [np.int32(enlargened_poly)], (color))

    padding = cv2.bitwise_and(padded_mask, padded_mask, mask=mask_inv)
    padding = np.expand_dims(padding, 2)
    padding = np.repeat(padding, 3, 2)

    padded_img = cv2.add(padding, extracted_img)

    return padded_img, padded_mask


def extract_and_pad_mask(fg_img, fg_mask, bg_mask, padding_ratio, transform=True):
    """
    use mask to extract image portion and pad with white border (set ratio to 0
    for no padding)

    uses cv2.findContours to locate separate polygons in a single mask, if more
    than one present. added padding may overlap, nothing is implemented
    to handle this.

    param fg_img: 3d opencv image
         fg_mask: 2d binary mask with masked pixels == 255
         padding_ratio: describes size of padded border added
         transform: whether or not to augment images and masks
    return results: list of dicts, one for each detected object in mask.
            each tuple contains an image, mask with padding and mask without padding
    """
    # threshold to make binary
    # if transform:
    #     tmp_fg_mask = np.zeros(fg_img.shape, dtype=np.uint8)
    #     fg_img, fg_mask = transforms(fg_img, fg_mask, bg_mask)
    #     fg_mask = fg_mask.draw_on_image(tmp_fg_mask)[0]
    #     print(fg_img.shape, fg_img.dtype, fg_mask.shape, fg_mask.dtype)

    _, threshold = cv2.threshold(fg_mask, 110, 255,
                            cv2.THRESH_BINARY)
    # find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    results = []
    for cnt in contours:
        # convert contour to polygon
        poly = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        # create new mask with only current polygon
        this_poly_mask = np.zeros(fg_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(this_poly_mask, [poly], (255))
        # enlargen polygon for padding
        enlargened_poly = np.squeeze(enlargen_polygon(poly, padding_ratio), axis=1)
        # get image of original polygon + added padding
        padded_poly_img, padded_mask = \
                get_padded_polygon_image(enlargened_poly, fg_img, this_poly_mask)
        # get indices to crop from original fg_img into smallest region possible
        min_y, max_y, min_x, max_x = get_crop_indices_from_mask(padded_mask)
        padded_poly_img = padded_poly_img[min_y:max_y,min_x:max_x,:]
        padded_mask = padded_mask[min_y:max_y,min_x:max_x]
        this_poly_mask = this_poly_mask[min_y:max_y, min_x:max_x]
        results.append({"padded_img":padded_poly_img,
            "padded_mask": padded_mask, "annotations_mask": this_poly_mask})

    return results

def randomly_choose_overlay_location(fg, bg_mask, step=50):
    """
    given a mask indicating positions of previous overlays, find a new position
    to overlay current fg

    param fg: foreground to overlay (typically padded)
          bg_mask: np array, binary mask, masked pixels == 255
    return: row, col of start point
    """
    # get height, width of img mask
    rows, cols  = bg_mask.shape
    # get height, with of current foreground to overlay
    h, w, _ = fg.shape
    # adjust max row, col by subtracting foreground dimensions
    rows = rows - h
    cols = cols - w
    # get list of possible starting coordinates
    possible_starting_points = list(product([i for i in range(0, rows, step)], [i for i in range(0, cols, step)]))
    starting_indices = [i for i in range(len(possible_starting_points))]

    # until a good region is found, randomly sample from possible overlay regions
    # and check to see if any previous overlays intersect with that position
    while len(starting_indices):
        start = np.random.choice(starting_indices, 1)[0]
        start = starting_indices.pop(start)
        row, col = possible_starting_points[start]
        slice = bg_mask[row:row+h, col:col+w]
        if slice.sum() == 0:
            return row, col

    return None

def overlay(start_coords, padded_fg_img, padded_fg_mask, fg_anno_mask, bg_img, bg_mask):
    """
    use padded foreground mask to overlay padded fg onto bg
    update the bg_mask with annotations mask of fg that is currently overlayed

    param: padded_fg_img, np array, 3d cv2 image depicting an object with padding
           padded_fg_mask: np array, binary mask for padded img with masked pixels == 255
           fg_anno_mask: np array, binary mask, correspond to object without padding, masked pixels ==255
           bg_img: np array, 3d cv2 image
           bg_mask: np array, 2d binary mask of same dims as bg, showing all objects overlayed
           up to now

    return: bg_img, updated bg_img with new object overlayed
            bg_mask, updated bg_mask with fg_anno_mask
            in new overlay
    """
    row, col = start_coords
    h, w = padded_fg_mask.shape

    # create new mask of same dims as bg and place padded_fg_mask there at proper location
    tmp_bg_mask = np.zeros(shape= bg_mask.shape, dtype=np.uint8)
    tmp_bg_mask[row:row+h, col:col+w] = padded_fg_mask
    tmp_bg_mask_inv = cv2.bitwise_not(tmp_bg_mask)

    # create new img of same dims as bg, place padded_fg_img there
    tmp_fg_img = np.zeros(bg_img.shape, dtype=np.uint8)
    tmp_fg_img[row:row+h, col:col+w] = padded_fg_img

    # use mask to combine bg_img, tmp_fg_img
    bg_img = cv2.bitwise_and(bg_img, bg_img, mask=tmp_bg_mask_inv)
    tmp_fg_img = cv2.bitwise_and(tmp_fg_img, tmp_fg_img, mask=tmp_bg_mask)
    bg_img = cv2.add(bg_img, tmp_fg_img)

    # update bg_mask with annos
    bg_mask[row:row+h, col:col+w] += fg_anno_mask

    return bg_img, bg_mask

def transforms(fg_img, fg_mask, bg_mask):
    bg_h, bg_w = bg_mask.shape
    resize_coeff = np.random.uniform(0.1,0.5)
    w = int(bg_w * resize_coeff)

    segmap = SegmentationMapsOnImage(fg_mask, shape=fg_mask.shape)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
                    iaa.Resize({"height":"keep-aspect-ratio", "width":w}),
                    sometimes(iaa.GaussianBlur(sigma=(0,3.0))),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01,0.1)))
                    ])

    fg_img, fg_mask = seq(image=fg_img, segmentation_maps=segmap)
    return fg_img, fg_mask



if __name__ == "__main__":

    img = "/home/benteau/Projects/advanced_barcode/tmp/0282925037198-01_N95-2592x1944_scaledTo640x480bilinear_rotation_65.jpg"
    mask = "/home/benteau/Projects/advanced_barcode/tmp/0282925037198-01_N95-2592x1944_scaledTo640x480bilinear_rotation_65.png"
    bg = "/home/benteau/Projects/advanced_barcode/synthetic_data/curated/background/top/unmapped/office-desk-table-supplies-top-view-.jpg"

    bg_img = cv2.imread(bg)
    img = cv2.imread(img)
    mask = cv2.imread(mask)[:,:,0]
    _, mask = cv2.threshold(mask, 110, 255,
                            cv2.THRESH_BINARY)

    bg_mask = np.zeros(bg_img.shape[:2], dtype=np.uint8)
    fg_data = extract_and_pad_mask(img, mask, bg_mask, 1.8)

    for item in fg_data:
        padded_fg_img, padded_fg_mask, fg_anno_mask = \
            item["padded_img"], item["padded_mask"], item["annotations_mask"]
        start = randomly_choose_overlay_location(padded_fg_img, bg_mask)

        if start == None:
            continue

        bg_img, bg_mask = overlay(start, padded_fg_img, padded_fg_mask,
                                fg_anno_mask, bg_img, bg_mask)
        plt.imshow(bg_img)
        plt.show()
        plt.imshow(bg_mask, cmap="gray")
        plt.show()
