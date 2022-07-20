HONEYCOMB
==============================

just another minimalist data science skeleton

Project Organization
------------

    ├── LICENSE
    ├── README.md                    <- The top-level README for developers using this project.
    ├── data
    │   ├── external                 <- Data from third party sources.
    │   ├── interim                  <- Intermediate data that has been transformed.
    │   ├── features                 <- Features extracted from raw data using domain knowledge.
    │   ├── processed                <- The final, canonical data sets for modeling (merge between interim and features).
    │   └── raw                      <- The original, immutable data dump.
    │
    ├── docs                         <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                       <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── references                   <- Data dictionaries, manuals, and all other explanatory materials.
    │
    │
    ├── requirements.txt             <- The requirements file for reproducing the analysis environment, e.g.
    │                                    generated with `pip freeze > requirements.txt`
    │ 
    ├── notebooks                    <- Jupyter notebooks (Source code for use in this project and EDA)
    │    │
    │    │
    │    └── visualization           <- Scripts to create exploratory and results oriented
    │
    └── src                          <- Main Python script (Source code for use in this project)
        │
        ├── __init__.py              <- Makes src a Python module
        │
        ├── data                     
        │   └── make_dataset         <- Scripts to download or generate data
        │   │         
        │   └── data_io              <- Functions to load and save data
        │   │         
        │   └── preprocess_tabular   <- Functions to preprocess raw tabular data
        │   │         
        │   └── preprocess_sound     <- Functions to preprocess sound or time series data
        │   │         
        │   └── preprocess_sound     <- Functions to preprocess text raw data
        │   │      
        │   └── preprocess_image     <- Functions to preprocess raw image data
        │
        │
        ├── features                 <- Scripts to turn raw data into features for modeling
        │   │         
        │   └── tabular_feature_engineer   <- Functions to make feature engineer from raw tabular data
        │   │         
        │   └── sound_feature_engineer     <- Functions to make feature engineer from raw data sound or time series data
        │   │         
        │   └── text_feature_engineer      <- Functions to make feature engineer from text data 
        │   │      
        │   └── build_features             <- main script from features eng. function are called 
        │
        │
        ├── models                   <- Scripts to train models and then use trained models to make predictions
        │   │                                               
        │   ├── run.sh               <- bash script where train and test experiment are setted up
        │   │ 
        │   └── train                <- main script to train model
        │   │ 
        │   └── inference            <- main script to inference from trained model   
        │   
        │   
        │   
        ├── fold                    <- Scripts to create folds from datasets 
        │   
        └── metric                  <- Scripts with many validation functions  

--------


# honeycomb

Honeycomb is a simple Machine learning framework with the aim to have a simple but useful starting point for data science projects and kaggle competitions. Some things have be borrow from other projects and have been adapted to it.
