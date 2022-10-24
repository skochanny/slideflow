'''Module for the `sf.Dataset` class and its associated functions.

The Dataset class handles management of collections of patients,
clinical annotations, slides, extracted tiles, and assembly of images
into torch DataLoader and tensorflow Dataset objects. The high-level
overview of the structure of the Dataset class is as follows:


 ──────────── Information Methods ───────────────────────────────
   Annotations      Slides        Settings         TFRecords
  ┌──────────────┐ ┌─────────┐   ┌──────────────┐ ┌──────────────┐
  │Patient       │ │Paths to │   │Tile size (px)│ | *.tfrecords  |
  │Slide         │ │ slides  │   │Tile size (um)│ |  (generated) |
  │Label(s)      │ └─────────┘   └──────────────┘ └──────────────┘
  │ - Categorical│  .slides()     .tile_px         .tfrecords()
  │ - Continuous │  .rois()       .tile_um         .manifest()
  │ - Time Series│  .slide_paths()                 .num_tiles
  └──────────────┘  .thumbnails()                  .img_format
    .patients()
    .rois()
    .labels()
    .harmonize_labels()
    .is_float()


 ─────── Filtering and Splitting Methods ──────────────────────
  ┌────────────────────────────┐
  │                            │
  │ ┌─────────┐                │ .filter()
  │ │Filtered │                │ .remove_filter()
  │ │ Dataset │                │ .clear_filters()
  │ └─────────┘                │ .train_val_split()
  │               Full Dataset │
  └────────────────────────────┘


 ───────── Summary of Image Data Flow ──────────────────────────
  ┌──────┐
  │Slides├─────────────┐
  └──┬───┘             │
     │                 │
     ▼                 │
  ┌─────────┐          │
  │TFRecords├──────────┤
  └──┬──────┘          │
     │                 │
     ▼                 ▼
  ┌────────────────┐ ┌─────────────┐
  │torch DataLoader│ │Loose images │
  │ / tf Dataset   │ │ (.png, .jpg)│
  └────────────────┘ └─────────────┘

 ──────── Slide Processing Methods ─────────────────────────────
  ┌──────┐
  │Slides├───────────────┐
  └──┬───┘               │
     │.extract_tiles()   │.extract_tiles(
     ▼                   │    save_tiles=True
  ┌─────────┐            │  )
  │TFRecords├────────────┤
  └─────────┘            │ .extract_tiles
                         │  _from_tfrecords()
                         ▼
                       ┌─────────────┐
                       │Loose images │
                       │ (.png, .jpg)│
                       └─────────────┘


 ─────────────── TFRecords Operations ─────────────────────────
                      ┌─────────┐
   ┌────────────┬─────┤TFRecords├──────────┐
   │            │     └─────┬───┘          │
   │.tfrecord   │.tfrecord  │ .balance()   │.resize_tfrecords()
   │  _heatmap()│  _report()│ .clip()      │.split_tfrecords
   │            │           │ .torch()     │  _by_roi()
   │            │           │ .tensorflow()│
   ▼            ▼           ▼              ▼
  ┌───────┐ ┌───────┐ ┌────────────────┐┌─────────┐
  │Heatmap│ │PDF    │ │torch DataLoader││TFRecords│
  └───────┘ │ Report│ │ / tf Dataset   │└─────────┘
            └───────┘ └────────────────┘
'''

from .dataset import Dataset
from .dataset_utils import (split_patients,
                            split_patients_balanced,
                            split_patients_preserved_site)
from .recipe import Recipe, build_local_recipe