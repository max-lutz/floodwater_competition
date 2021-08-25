import sys
from pathlib import Path

from loguru import logger
import numpy as np
import typer
from tifffile import imwrite, imread
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
import rasterio

ROOT_DIRECTORY = Path("/codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"


def make_predictions(chip_id: str, model):
    """
    Given an image ID, read in the appropriate files and predict a mask of all ones or zeros
    """
    try:
        arr_vh = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif")
        arr_vv = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif")
        
        #load image from path
        with rasterio.open(arr_vv) as vv:
            vv_path = vv.read(1)
        with rasterio.open(arr_vh) as vh:
            vh_path = vh.read(1)
        
        img = np.stack([vv_path, vh_path], axis=-1)
        output_prediction = model.predict(img)

    except:
        logger.warning(
            f"test_features not found for {chip_id}, predicting all zeros; did you download your"
            f"training data into `runtime/data/test_features` so you can test your code?"
        )
        output_prediction = np.zeros(shape=(512, 512), dtype=np.uint8)
    return output_prediction


def get_expected_chip_ids():
    """
    Use the input directory to see which images are expected in the submission
    """
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    # images are named something like abc12.tif, we only want the abc12 part
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    return ids


def main():
    """
    for each input file, make a corresponding output file using the `make_predictions` function
    """
    logger.info("Loading model")
    model = keras.models.load_model(ASSETS_DIRECTORY / "model_floodwater_unet_basic.h5")


    logger.info("Finding chip IDs")
    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)
    
    logger.info(f"found {len(chip_ids)} expected image ids; generating predictions for each ...")
    for chip_id in tqdm(chip_ids, miniters=25, file=sys.stdout, leave=True):
        # figure out where this prediction data should go
        output_path = SUBMISSION_DIRECTORY / f"{chip_id}.tif"
        # make our predictions! (you should edit `make_predictions` to do something useful)
        output_data = make_predictions(chip_id, model)
        imwrite(output_path, output_data, dtype=np.uint8)
    logger.success(f"... done")


if __name__ == "__main__":
    typer.run(main)
