import sys
from loguru import logger
import numpy as np
import typer
from tifffile import imwrite
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import os

import tensorflow as tf
from tensorflow import keras
import rasterio


def make_predictions(chip_id: str, model):
    """
    Given an image ID, read in the appropriate files and predict a mask of all ones or zeros
    """
    #logger.info(os.path.join(os.getcwd(), 'data', 'test_features', chip_id+'_vh.tif'))
    try:
        #os.path.abspath()
        path_vh = os.path.join(os.getcwd(), 'data', 'test_features', chip_id+'_vh.tif')
        path_vv = os.path.join(os.getcwd(), 'data', 'test_features', chip_id+'_vv.tif')
        # arr_vh = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif")
        # arr_vv = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif")
        
        #load image from path
        with rasterio.open(path_vh) as vv:
            arr_vh = vv.read(1)
        with rasterio.open(path_vv) as vh:
            arr_vv = vh.read(1)
        
        img = np.array([np.stack([arr_vv, arr_vh], axis=-1)])

        #logger.info(img.shape)

        #config = model.get_config() # Returns pretty much every information about your model
        #logger.info(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels

        output_prediction = model.predict(img)[0,:, :, 0]

        output_prediction = ((output_prediction > 0.5) * 1).astype(np.uint8)

        logger.info(output_prediction.shape)

    except:
        logger.warning(
            f"test_features not found for {chip_id}, predicting all zeros; did you download your"
            f"training data into `runtime/data/test_features` so you can test your code?"
        )
        output_prediction = np.zeros(shape=(512, 512))
    return output_prediction


def get_expected_chip_ids():
    """
    Use the input directory to see which images are expected in the submission
    """
    mypath = os.path.join(os.getcwd(), 'data', 'test_features')
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    
    #logger.info(files)
    # images are named something like abc12.tif, we only want the abc12 part
    ids = list(sorted(set(f.split("_")[0] for f in files)))
    #logger.info(ids)
    return ids


def main():
    """
    for each input file, make a corresponding output file using the `make_predictions` function
    """
    logger.info("Loading model")
    model = keras.models.load_model(os.path.join(os.getcwd(), 'assets', 'model_floodwater_unet_basic.h5'))
    #logger.info(model.summary())


    logger.info("Finding chip IDs in ")
    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)
    
    logger.info(f"found {len(chip_ids)} expected image ids; generating predictions for each ...")
    for chip_id in tqdm(chip_ids, miniters=25, file=sys.stdout, leave=True):
        # figure out where this prediction data should go
        output_path = os.path.join(os.getcwd(), 'submission', chip_id+'.tif')
        # make our predictions! (you should edit `make_predictions` to do something useful)
        output_data = make_predictions(chip_id, model)
        imwrite(output_path, output_data, dtype=np.uint8)
    logger.success(f"... done")


if __name__ == "__main__":
    typer.run(main)
