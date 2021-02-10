HONEYCOMB
==============================

just another minimalist data science skeleton

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── features       <- Features extracted from raw data using domain knowledge.
    │   ├── processed      <- The final, canonical data sets for modeling (merge between interim and features).
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │ 
    ├── notebooks          <- Jupyter notebooks (Source code for use in this project)
    │    │
    │    └── visualization  <- Scripts to create exploratory and results oriented
    │
    └── src               <- Jupyter notebooks (Source code for use in this project)
        │
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.ipynb
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.ipynb
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── run.sh     <- bash script where train and test experiment are setted up
        │   │
        │   │
        │   ├── train_model.py    <-
        │   │ 
        │   ├── predict_model.py  <-
        │   │
        │   ├── train_cvmodel.py  <-
	│   │
        │   ├── random_search.py  <-
        │   │
        │   ├── metric_dispatcher.py <-
        │   │
        │   └── model_dispatcher.py  <-
        │
        │
        └── fold  <- Scripts to create folds from datasets 

--------


# honeycomb
