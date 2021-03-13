from diffsims.utils import mask_utils as mu
import numpy as np


def test_create_mask():
    mask = mu.create_mask((20, 10))
    assert mask.shape[0] == 20
    assert mask.shape[1] == 10


def test_invert_mask():
    mask = mu.create_mask((20, 10))
    initial = mask[0,0]
    mu.invert_mask(mask)
    assert initial != mask[0,0]


def test_add_polygon():
    mask = mu.create_mask((20, 10))
    coords = np.array([[5, 5],[15, 5],[10,10]])
    mu.add_polygon_to_mask(mask, coords)


def test_add_circles_to_mask():
    mask = mu.create_mask((20, 10))
    coords = np.array([[5, 5],[15, 5],[10,10]])
    mu.add_circles_to_mask(mask, coords, 3)
    

def test_add_circle_to_mask():
    mask = mu.create_mask((20, 10))
    mu.add_circle_to_mask(mask, 5, 5, 5)


def test_add_annulus_to_mask():
    mask = mu.create_mask((20, 10))
    mu.add_annulus_to_mask(mask, 4, 7)


def test_add_band_to_mask():
    mask = mu.create_mask((20, 10))
    mu.add_band_to_mask(mask, 4, 7, 10, 4)
