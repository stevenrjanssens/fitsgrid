import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from cyclopts import App
import matplotlib.pyplot as plt
from matplotlib import colors


app = App(help="Create 3x3 cutout grids to evaluate image quality.")

cutout_size = (500, 500) # (width, height) in pixels

symlognorm = colors.SymLogNorm(linthresh=10, linscale=1, vmin=-50, vmax=1<<16)


def gridimage(path: str):
    """
    Create 3x3 grid cutout image for FITS file specified by path.

    Parameters
    ----------
    path : str
        Path to FITS image.
    """
    data = fits.getdata(path)
    img_height, img_width = data.shape

    # Define positions for 3x3 grid cutouts
    positions = [
        (cutout_size[0] // 2, img_height - cutout_size[1] // 2),  # Top-left
        (img_width // 2, img_height - cutout_size[1] // 2),  # Top-center
        (img_width - cutout_size[0] // 2, img_height - cutout_size[1] // 2),  # Top-right

        (cutout_size[0] // 2, img_height // 2),  # Left-center
        (img_width // 2, img_height // 2),  # Center
        (img_width - cutout_size[0] // 2, img_height // 2),  # Right-center

        (cutout_size[0] // 2, cutout_size[1] // 2),  # Bottom-left
        (img_width // 2, cutout_size[1] // 2),  # Bottom-center
        (img_width - cutout_size[0] // 2, cutout_size[1] // 2)  # Bottom-right
    ]

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, pos in zip(axes.flat, positions):
        cutout = Cutout2D(data, pos, cutout_size)
        ax.imshow(cutout.data - np.median(cutout.data), cmap="gray", norm=symlognorm, origin="lower")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


@app.default
def main(*paths: str):
    """
    Creates a 3x3 cutout grid for each FITS file in paths list.

    Parameters
    ----------
    paths : str
        List of FITS files to create cutout grids for.
    """
    for path in paths:
        gridimage(path)


if __name__ == '__main__':
    app()
