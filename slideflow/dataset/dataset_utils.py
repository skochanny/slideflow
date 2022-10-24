"""Utility functions for the Dataset class."""

import shutil
import time
from os.path import basename,join
from queue import Queue
from random import shuffle
from typing import Dict, List, Optional, Sequence
import pandas as pd

import slideflow as sf
from slideflow import errors
from slideflow.util import log, _shortname, path_to_name


def _prepare_slide(
    path: str,
    report_dir: Optional[str],
    tma: bool,
    wsi_kwargs: Dict,
    qc: Optional[str],
    qc_kwargs: Dict,
) -> Optional[sf.slide._BaseLoader]:

    try:
        if tma:
            slide = sf.TMA(
                path=path,
                tile_px=wsi_kwargs['tile_px'],
                tile_um=wsi_kwargs['tile_um'],
                stride_div=wsi_kwargs['stride_div'],
                enable_downsample=wsi_kwargs['enable_downsample'],
                report_dir=report_dir
            )  # type: sf.slide._BaseLoader
        else:
            slide = sf.WSI(path, **wsi_kwargs)
        if qc:
            slide.qc(method=qc, **qc_kwargs)
        return slide
    except errors.MissingROIError:
        log.debug(f'Missing ROI for slide {path}; skipping')
        return None
    except errors.SlideLoadError as e:
        log.error(f'Error loading slide {path}: {e}. Skipping')
        return None
    except errors.QCError as e:
        log.error(e)
        return None
    except errors.TileCorruptionError:
        log.error(f'{path} corrupt; skipping')
        return None
    except (KeyboardInterrupt, SystemExit) as e:
        print('Exiting...')
        raise e


def _tile_extractor(
    path: str,
    tfrecord_dir: str,
    tiles_dir: str,
    reports: Dict,
    tma: bool,
    qc: str,
    wsi_kwargs: Dict,
    generator_kwargs: Dict,
    qc_kwargs: Dict
) -> None:
    """Internal function to extract tiles. Slide processing needs to be
    process-isolated when num_workers > 1 .

    Args:
        tfrecord_dir (str): Path to TFRecord directory.
        tiles_dir (str): Path to tiles directory (loose format).
        reports (dict): Multiprocessing-enabled dict.
        tma (bool): Slides are in TMA format.
        qc (bool): Quality control method.
        wsi_kwargs (dict): Keyword arguments for sf.WSI.
        generator_kwargs (dict): Keyword arguments for WSI.extract_tiles()
        qc_kwargs(dict): Keyword arguments for quality control.
    """
    try:
        log.debug(f'Extracting tiles for {path_to_name(path)}')
        slide = _prepare_slide(
            path,
            report_dir=tfrecord_dir,
            tma=tma,
            wsi_kwargs=wsi_kwargs,
            qc=qc,
            qc_kwargs=qc_kwargs)
        if slide is not None:
            report = slide.extract_tiles(
                tfrecord_dir=tfrecord_dir,
                tiles_dir=tiles_dir,
                **generator_kwargs
            )
            reports.update({path: report})
    except errors.MissingROIError:
        log.info(f'Missing ROI for slide {path}; skipping')
    except errors.SlideLoadError as e:
        log.error(f'Error loading slide {path}: {e}. Skipping')
    except errors.QCError as e:
        log.error(e)
    except errors.TileCorruptionError:
        log.error(f'{path} corrupt; skipping')
    except (KeyboardInterrupt, SystemExit) as e:
        print('Exiting...')
        raise e


def _fill_queue(
    slide_list: Sequence[str],
    q: Queue,
    q_size: int,
    buffer: Optional[str] = None
) -> None:
    '''Fills a queue with slide paths, using an optional buffer.'''
    for path in slide_list:
        warned = False
        if buffer:
            while True:
                if q.qsize() < q_size:
                    try:
                        buffered = join(buffer, basename(path))
                        shutil.copy(path, buffered)
                        q.put(buffered)
                        break
                    except OSError:
                        if not warned:
                            slide = _shortname(path_to_name(path))
                            log.debug(f'OSError for {slide}: buffer full?')
                            log.debug(f'Queue size: {q.qsize()}')
                            warned = True
                        time.sleep(1)
                else:
                    time.sleep(1)
        else:
            q.put(path)
    q.put(None)
    q.join()


def _count_otsu_tiles(wsi):
    wsi.qc('otsu')
    return wsi.estimated_num_tiles


def split_patients_preserved_site(
    patients_dict: Dict[str, Dict],
    n: int,
    balance: str,
    method: str = 'auto'
) -> List[List[str]]:
    """Splits a dictionary of patients into n groups,
    balancing according to key "balance" while preserving site.

    Args:
        patients_dict (dict): Nested dictionary mapping patient names to
            dict of outcomes: labels
        n (int): Number of splits to generate.
        balance (str): Annotation header to balance splits across.
        method (str): Solver method. 'auto', 'cplex', or 'bonmin'. If 'auto',
            will use CPLEX if availabe, otherwise will default to pyomo/bonmin.

    Returns:
        List of patient splits
    """
    patient_list = list(patients_dict.keys())
    shuffle(patient_list)

    def flatten(arr):
        '''Flattens an array'''
        return [y for x in arr for y in x]

    # Get patient outcome labels
    patient_outcome_labels = [
        patients_dict[p][balance] for p in patient_list
    ]
    # Get unique outcomes
    unique_labels = list(set(patient_outcome_labels))
    n_unique = len(set(unique_labels))
    # Delayed import in case CPLEX not installed
    import slideflow.io.preservedsite.crossfolds as cv

    site_list = [patients_dict[p]['site'] for p in patient_list]
    df = pd.DataFrame(
        list(zip(patient_list, patient_outcome_labels, site_list)),
        columns=['patient', 'outcome_label', 'site']
    )
    df = cv.generate(
        df, 'outcome_label', k=n, target_column='CV', method=method
    )
    log.info("[bold]Train/val split with Preserved-Site Cross-Val")
    log.info("[bold]Category\t" + "\t".join(
        [str(cat) for cat in range(n_unique)]
    ))
    for k in range(n):
        def num_labels_matching(o):
            match = df[(df.CV == str(k+1)) & (df.outcome_label == o)]
            return str(len(match))
        matching = [num_labels_matching(o) for o in unique_labels]
        log.info(f"K-fold-{k}\t" + "\t".join(matching))
    splits = [
        df.loc[df.CV == str(ni+1), "patient"].tolist()
        for ni in range(n)
    ]
    return splits


def split_patients_balanced(
    patients_dict: Dict[str, Dict],
    n: int,
    balance: str
) -> List[List[str]]:
    """Splits a dictionary of patients into n groups,
    balancing according to key "balance".

    Args:
        patients_dict (dict): Nested ditionary mapping patient names to
            dict of outcomes: labels
        n (int): Number of splits to generate.
        balance (str): Annotation header to balance splits across.

    Returns:
        List of patient splits
    """
    patient_list = list(patients_dict.keys())
    shuffle(patient_list)

    def flatten(arr):
        '''Flattens an array'''
        return [y for x in arr for y in x]

    # Get patient outcome labels
    patient_outcome_labels = [
        patients_dict[p][balance] for p in patient_list
    ]
    # Get unique outcomes
    unique_labels = list(set(patient_outcome_labels))
    n_unique = len(set(unique_labels))

    # Now, split patient_list according to outcomes
    pt_by_outcome = [
        [p for p in patient_list if patients_dict[p][balance] == uo]
        for uo in unique_labels
    ]
    # Then, for each sublist, split into n components
    pt_by_outcome_by_n = [
        list(sf.util.split_list(sub_l, n)) for sub_l in pt_by_outcome
    ]
    # Print splitting as a table
    log.info(
        "[bold]Category\t" + "\t".join([str(cat) for cat in range(n_unique)])
    )
    for k in range(n):
        matching = [str(len(clist[k])) for clist in pt_by_outcome_by_n]
        log.info(f"K-fold-{k}\t" + "\t".join(matching))
    # Join sublists
    splits = [
        flatten([
            item[ni] for item in pt_by_outcome_by_n
        ]) for ni in range(n)
    ]
    return splits


def split_patients(patients_dict: Dict[str, Dict], n: int) -> List[List[str]]:
    """Splits a dictionary of patients into n groups."

    Args:
        patients_dict (dict): Nested ditionary mapping patient names to
            dict of outcomes: labels
        n (int): Number of splits to generate.

    Returns:
        List of patient splits
    """
    patient_list = list(patients_dict.keys())
    shuffle(patient_list)
    return list(sf.util.split_list(patient_list, n))
