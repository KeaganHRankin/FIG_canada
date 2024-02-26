# The Future Infrastructure Growth Model - Canadian Application (FIG Canada)
![alt text](https://github.com/KeaganHRankin/FIG_canada/blob/main/graphics/fig_canada_logo.JPG)
## Main
FIG-Canada is a model for forecasting and investigating embodied emissions from future housing and infastructure construction. See our [preprint here](https://dx.doi.org/10.2139/ssrn.4647023) for more information around the model and this implementation for Canada.
![alt text](https://github.com/KeaganHRankin/FIG_canada/blob/main/graphics/table%20of%20contents%20graphic.png)

## Data and Inputs
-where to get data and potentially post modified data to Zenodo

## Model
The model/fig_package folder contains the main utilities used to run FIG. `helper.py` contains miscellaneous utils. `project_starts.py` returns projected housing starts for Canada. The `fig_sample` folder contains all of the functions for generating new/future neighbourhood layouts using monte carlo sampling; specifically, `sample.py` contains the aggregate functions to sample all infrastructure and housing. `forecast.py` contains generic functions for forecasting a future year's construction. The two notebooks in the file show how the model can be applied to generated new neighbourhoods and corresponding embodied emissions. Note that file-paths in function initializations will need to be changed if run locally.

## Projections
-projecting starts and mat changes

## Results
-the result notebooks

## Contributions
-

## References
-data papers and our papers
