"""Loading and checking the pork rasher packaging dataset."""

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images(folder):
    """Count the image files in a folder.

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory containing image files.

    Returns
    -------
    int
        Number of files with a .jpg, .jpeg or .png extension
        (case-insensitive).

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist or is not a directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)