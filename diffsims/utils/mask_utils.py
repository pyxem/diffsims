from PIL import ImageDraw
from PIL import Image
import numpy as np


def create_mask(shape, fill=True):
    """
    Instantiate an empty mask
    """
    return np.full(shape, fill, dtype=np.bool)


def invert_mask(mask):
    """
    Turn True into False and False into True
    """
    mask[:] = np.invert(mask)


def add_polygon_to_mask(coords, mask, fill=False):
    """
    Add a poligon defined by sequential vertex coordinates to the mask.

    Parameters
    ----------
    coords: (N, 2) array
        (x, y) coordinates of vertices
    mask: (H, W) array of dtype bool
        boolean mask for an image
    fill: int, optional
        Fill value. 0 is black (negative, False) and 1 is white (True)
    """
    coords = np.array(coords)
    coords = np.ravel(coords, order='C').tolist()
    tempmask = Image.fromarray(mask)
    draw = ImageDraw.Draw(tempmask)
    draw.polygon(coords, fill=fill)
    mask[:] = np.array(tempmask, dtype=np.bool)


def add_circles_to_mask(coords, r, mask, fill=False):
    """
    Add a circle on a mask at each (x, y) coordinate with a radius r

    Parameters
    ----------
    coords: (N, 2) array
        (x, y) coordinates of circle centers
    r: float or (N,) array
        radii of the circles
    mask: (H, W) array of dtype bool
        boolean mask for an image
    fill: int, optional
        Fill value. 0 is black (negative, False) and 1 is white (True)
    """
    coords = np.array(coords)
    r = r*np.ones(coords.shape[0])
    for i, j in zip(coords, r):
        add_circle_to_mask(i[0], i[1], j, mask, fill=fill)


def add_circle_to_mask(x, y, r, mask, fill=False):
    """
    Add a single circle to the mask

    Parameters
    ----------
    x: float
        x-coordinate of the circle center in pixels
    y: float
        y-coordinate of the circle center in pixels
    r: float
        radius of the circles in pixels
    mask: (H, W) array of dtype bool
        boolean mask for an image
    fill: int, optional
        Fill value. 0 is black (negative, False) and 1 is white (True)

    Returns
    -------
    mask: (H, W) array of dtype bool
        The array with the new addition
    """
    xx = np.arange(mask.shape[1])
    yy = np.arange(mask.shape[0])
    X, Y = np.meshgrid(xx, yy)
    condition = ((X - x)**2 + (Y - y)**2 < r**2)
    mask[condition] = fill


def add_annulus_to_mask(r1, r2, mask, x=None, y=None, fill=False):
    """
    Add an annular feature on the mask

    Parameters
    ----------
    r1: float
        radius of the inner circle in pixels
    r2: float
        radius of the outer circle in pixels
    mask: (H, W) array of dtype bool
        boolean mask for an image
    x: float
        x-coordinate of the circle center in pixels. Defaults to the center of the mask.
    y: float
        y-coordinate of the circle center in pixels. Defaults to the center of the mask.
    fill: int, optional
        Fill value. 0 is black (block, False) and 1 is white (pass, True)
    """
    if x is None:
        x = mask.shape[1] / 2
    if y is None:
        y = mask.shape[0] / 2
    xx = np.arange(mask.shape[1])
    yy = np.arange(mask.shape[0])
    X, Y = np.meshgrid(xx, yy)
    condition = ((X - x)**2 + (Y - y)**2 > r1**2) & ((X - x)**2 + (Y - y)**2 < r2**2)
    mask[condition] = fill
