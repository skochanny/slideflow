![slideflow logo](https://github.com/jamesdolezal/slideflow/raw/master/docs-source/pytorch_sphinx_theme/images/slideflow-banner.png)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5703792.svg)](https://doi.org/10.5281/zenodo.5703792)
[![Python application](https://github.com/jamesdolezal/slideflow/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/jamesdolezal/slideflow/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/slideflow.svg)](https://badge.fury.io/py/slideflow)

Slideflow is a computational pathology Python package which provides a unified API for building and testing deep learning models for histopathology, supporting both Tensorflow/Keras and PyTorch backends. 

Slideflow includes tools for **whole-slide image processing** and segmentation, **customizable deep learning model training** with dozens of supported architectures, **explainability tools** including heatmaps and mosaic maps, **analysis of activations** from model layers, **uncertainty quantification**, and more. A variety of fast, optimized whole-slide image processing tools are included, including background filtering, blur/artifact detection, digital stain normalization, and efficient storage in `*.tfrecords` format. Model training is easy and highly configurable, with support for dozens of model architectures (from `tf.keras.applications` or `torchvision.models`) and an easy drop-in API for training custom architectures. For entirely custom training loops, Slideflow can be used as an image processing backend, serving an optimized `tf.data.Dataset` or `torch.utils.data.DataLoader` which can read and process `*.tfrecords` images and perform real-time stain normalization.

Slideflow has been used by:

- [Dolezal et al](https://www.nature.com/articles/s41379-020-00724-3), _Modern Pathology_, 2020
- [Rosenberg et al](https://ascopubs.org/doi/10.1200/JCO.2020.38.15_suppl.e23529), _Journal of Clinical Oncology_ [abstract], 2020
- [Howard et al](https://www.nature.com/articles/s41467-021-24698-1), _Nature Communications_, 2021
- [Dolezal et al](https://arxiv.org/abs/2204.04516) [arXiv], 2022
- Partin et al [arXiv], 2022

## Installation
Slideflow requires Python 3.7+ and [libvips](https://libvips.github.io/libvips/) 8.9+.

Ensure you have the latest version of pip, setuptools, and wheel installed:

```
pip3 install --upgrade setuptools pip wheel
```

Install package requirements from source/requirements.txt:

```
pip3 install -r requirements.txt
```

Finally, install using pip:

```
pip3 install slideflow
```

## Getting started
Import the module in python and initialize a new project:

```python
import slideflow as sf
P = sf.Project.from_prompt("/path/to/project/directory")
```

You will be taken through a set of questions to configure your new project. Slideflow projects require an annotations file (CSV) associating patient names to outcome categories and slide names. If desired, a blank file will be created for you when you first setup a new project. Once the project is created, add rows to the annotations file with patient names and outcome categories.

Next, you will be taken through a set of questions to configure your first dataset source. Alternatively, you may manually add a source by calling:

```python
P.add_source(
  name="NAME",
  slides="/slides/directory",
  roi="/roi/directory",
  tiles="/tiles/directory",
  tfrecords="/tfrecords/directory"
)
```

Once your annotations file has been set up and you have a dataset to work with, begin extracting tiles at specified pixel and micron size:

```python
P.extract_tiles(tile_px=299, tile_um=302)
```

Following tile extraction, configure a set of model parameters:

```python
params = sf.model.ModelParams(
  tile_px=299,
  tile_um=302,
  batch_size=32,
  model='xception'
)
```

...and begin training:

```python
P.train('category1', params=params)
```

For complete documentation of all pipeline functions and example tutorials, please see the documentation at [slideflow.dev](https://www.slideflow.dev/).
