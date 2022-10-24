"""Builds, loads, and cooks recipes for reproducing Datasets."""

import uuid
import hashlib
import pandas as pd
from typing import Any, Union, List, Optional, Dict, Tuple
from rich.progress import track

import slideflow as sf
from slideflow.util import log
from slideflow.dataset import Dataset

# -----------------------------------------------------------------------------

def _checksum(path: str) -> str:
    """Calculate and return MD5 checksum for file.

    Args:
        path (str): Path to file to verify.

    Returns:
        str: MD5 checksum.
    """
    m = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(4096):
            m.update(chunk)
    return m.hexdigest()


def build_local_recipe(
    dataset: Dataset,
    name: str,
    description: str,
    extract_tiles_kwargs: Optional[Dict[str, Any]] = None,
    annotations: Optional[str] = None,
    checksum: bool = True
) -> Recipe:
    """Builds a local recipe from an existing Dataset.

    Args:
        dataset (Dataset, optional): Slideflow Dataset from which to build
            the recipe.
        name (str): Name of the recipe.
        description (str): Description of the recipe.
        extract_tiles_kwargs (dict, optional): Keyword arguments for tile
            extraction. If provided, must contain ``tile_px`` and ``tile_um``
            at minimum. Defaults to None (tiles not extracted).
        annotations (str, optional): Path to annotations file. Defaults to None.
        checksum (bool): Calculate and include checksum

    Returns:
        Recipe
    """
    log.debug(f'Building recipe "{name}"...')

    if checksum:
        slides = {}
        for slide in track(dataset.slide_paths(),
                           description="Calculating checksums..."):
            slides[sf.util.path_to_name(slide)] = {
                'path': slide,
                'md5': _checksum(slide)
            }
    else:
        slides = dataset.slide_paths()  # type: ignore

    return Recipe(
        name=name,
        description=description,
        uuid=uuid.uuid4().hex,
        slides=slides,
        rois=dataset.rois(),
        extract_tiles_kwargs=extract_tiles_kwargs,
        annotations=annotations,
    )


def load_slide_from_path(dest: str, path: str, md5: Optional[str]):
    """Load a slide from a path to a target destination.

    Args:
        dest (str): Destination path to save the slide.
        path (str): Path to slide to load, either local or remote.
        md5 (str, optional): MD5 checksum for slide.

    Returns:
        True if slide loaded successfully, False if the slide failed to load
        or failed the checksum integrity verification.
    """
    raise NotImplementedError


def load_slide_from_tcga(dest: str, tcga_uuid: str, md5: Optional[str]):
    """Download a slide from TCGA using its UUID.

    Args:
        dest (str): Destination path to save the slide.
        path (str): TCGA UUID for the slide.
        md5 (str, optional): MD5 checksum for slide.

    Returns:
        True if slide downloaded successfully, False if the slide failed to
        download or failed the checksum integrity verification.
    """
    raise NotImplementedError


def load_slide(dest: str, **kwargs) -> bool:
    """Load a slide using a given set of keyword arguments."""
    if 'path' in kwargs:
        return load_slide_from_path(dest, **kwargs)
    elif 'tcga_uuid' in kwargs:
        return load_slide_from_tcga(dest, **kwargs)
    else:
        raise ValueError(
            "Unable to determine slide loading method from keyword arguments. "
            "Expected either 'path' or 'tcga_uuid'. Got: {}".format(
                ', '.join(list(kwargs.keys())))
            )


def slide_directory_tree(
    path: str
) -> Tuple[Dict[str, str],
           Optional[Dict[str, List[str]]]]:
    """Recursively search a directory for slides, building a directory tree.

    Args:
        path (str): Directory to search.

    Returns:
        A tuple containing:
            dict: Dictionary mapping slide names to paths for all discovered
            slides, exclusive of duplicate slides.

            dict, optional: Dictionary mapping names of slides with duplicate
            paths, to list containing path to duplicates.
    """
    return dict(), None


# -----------------------------------------------------------------------------

class Recipe:

    def __init__(
        self,
        name: str,
        description: str,
        uuid: str,
        slides: Union[List[str], Dict[str, Dict[str, Any]], str],
        rois: Optional[Union[List[str], str]] = None,
        extract_tiles_kwargs: Optional[Dict[str, Any]] = None,
        annotations: Optional[str] = None
    ) -> None:
        """Recipe for constructing a Dataset and optionally extracting tiles.

        Args:
            name (str): Human-readable recipe name.
            description (str): Description of the recipe, how it is used, etc.
            uuid (str): Unique numeric identifier, as generated by uuid4().hex
            slides (Union[List[str], List[Dict[str, Any]], str]): Whole slides.
                ``str``: Path to a .tar.gz file with slides, local or remote.
                ``list(str)``: List of slide paths, local or remote.
                ``dict``: Dictionary mapping slide names/identifiers to
                dictionary keyword arguments
            rois (Union[List[str], str]): Regions of interest (CSV files).
                ``str``: Path to .tar.gz file with CSV files, local or remote.
                ``list(str)``: List of individual CSV paths, local or remote.
                Defaults to None.
            extract_tiles_kwargs (Dict[str, Any], optional): Keyword arguments
                for extracting tiles after all slides are loaded. If provided,
                must include ``tile_px`` and ``tile_um`` keys at minimum.
                All other keys are passed to ``Dataset.extract_tiles()``.
                Defaults to None.
            annotations (str, optional): Path to annotations file.
                Defaults to None.
        """
        self.name = name
        self.description = description
        self.uuid = uuid
        self.rois = rois
        self.slides = slides
        self.extract_tiles_kwargs = extract_tiles_kwargs
        self.annotations = annotations

    def __repr__(self):
        r =    "<Recipe("
        r += "\n   name: {!r}".format(self.name)
        r += "\n   description: {!r}".format(self.description)
        r += "\n   uuid: {!r}".format(self.uuid)
        r += "\n   extract_tiles_kwargs: {!r}".format(self.extract_tiles_kwargs)
        r += "\n   annotations: {!r}".format(self.annotations)
        if isinstance(self.slides, str):
            r += "\n   slides: {!r}".format(self.slides)
        else:
            r += "\n   slides: <{} total>".format(len(self.slides))
        if isinstance(self.rois, str):
            r += "\n   rois: {!r}".format(self.rois)
        else:
            r += "\n   rois: <{} total>".format(len(self.rois))
        r += "\n)>"
        return r

    def as_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'uuid': self.uuid,
            'extract_tiles_kwargs': self.extract_tiles_kwargs,
            'annotations': self.annotations,
            'slides': self.slides,
            'rois': self.rois
        }

    def to_json(self, path: str) -> None:
        sf.util.write_json(self.as_dict(), path)

    def make(self) -> None:
        pass