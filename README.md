# floodwater_competition

### Presentation of my model
I use three UNET models trained on three different split of the training data and take the average of their output.

### Installation steps

1. Create a new environment from environment.yml (it should contain all the requirements)
2. You can also install requirements from requirements.txt if needed.

### Files
Notebooks:
- EDAi.ipynb : perform simple EDA on the training data
- check_prediction.ipynb : Calculate the Jaccard score between the prediction (in the output_data folder) and the label data in /data/raw/train_features/train_labels. For this to work you need to make a prediction based on the .tif files in train_features.
It was a way for me to test my model before submitting it to the Drivendata platform.

Also there is some .gitkeep files in the empty folders to keep track of the folder while using git. You can ingore or remove them.
They should not affect the code.


### Execution steps for training a new model

1. Add training data in the folder data/raw/train_features, this training data should be in the same format as the one given during the competition:

    |-- data
        |-- raw
            |-- train_features
                |-- train_features                          <-- folder containing vv and vh .tif files
                |-- train_labels                            <-- folder containing the corresponding .tif label data
                |-- flood_training_metadata.csv             <-- csv file containing the metadata


2. Load the data from the planetary computer use the /src/data/load_pc_train_data.ipynb notebook
3. Train a model using /src/models/train_model.ipynb notebook
    3.1 In the notebook you can modify the train/test split, you can also modify the name of the model saved.
4. The model will be saved in /models/temporary/


### Execution steps for predicting new data

1. Add vv and vh .tif files to predict in the folder data/to_predict/test_features
2. Run the notbook /src/data/load_pc_test_data.ipynb to load the nasadem and jrc data from the planetary computer
2. Run the python code predict.py from src/models (command: python predict.py)
    2.1 As of right now the notebook is set up to make a prediction from the three models in the models folder and to output the average of their prediction.
3. The predicted images will be stored in /output_data/


Execution time of training:

Google colab
- CPU (model): single core hyper threaded Xeon Processors @2.3Ghz i.e(1 core, 2 threads)
- GPU (model or N/A): N/A
- Memory (GB): 12 Go
- OS: 
- Train duration: 30 min
- Inference duration: not done on google colab

Execution time of inference:

- CPU (model): AMD Ryzen 5 4500U
- GPU (model or N/A): N/A
- Memory (GB): 8 Go
- OS: WIndows 10
- Train duration: not trained on my computer
- Inference duration: 9 min 20 sec