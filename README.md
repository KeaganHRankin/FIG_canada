# The Future Infrastructure Growth Model - Canadian Application (FIG Canada)
![alt text](https://github.com/KeaganHRankin/FIG_canada/blob/main/graphics/fig_canada_logo.JPG)
## Main
FIG-Canada is a model for forecasting and investigating embodied emissions from future housing and infastructure construction. See our [preprint here](https://dx.doi.org/10.2139/ssrn.4647023) for more information around the model and this implementation for Canada.
![alt text](https://github.com/KeaganHRankin/FIG_canada/blob/main/graphics/table%20of%20contents%20graphic.png)

## Data and Inputs
The `data.txt` file describes the necessary inputs for each part of the model and provides a source where possible. Some data inputs were modified from open data in ArcGIS by the authors and is too large for this repository, but it can be re-created or provided upon reasonable request.

## Model
The model/fig_package folder contains the main utilities used to run FIG. `helper.py` contains miscellaneous utils. `project_starts.py` returns projected housing starts for Canada (sourced from CMHC estimates). The fig_sample folder contains all of the functions for generating new/future neighbourhood layouts using monte carlo sampling; specifically, `sample.py` contains the aggregate functions to sample all infrastructure and housing. `forecast.py` contains generic functions for forecasting a future year's construction. The two notebooks in the file show how the model can be applied to generated new neighbourhoods and corresponding embodied emissions. Note that file-paths in function initializations will need to be changed if run locally.

## Results
The result folder contains notebooks used to generate the figures and tables presented in the paper. Raw results are too large for repo but can be provided upon reasonable request. 

## Contributions
We are open to questions, concerns, and suggesting regarding FIG and its re-creation for other parts of the world.

## References and other relevant work
Housing data is described in this paper and can be found in the associated Zenodo repository: Guven, G.; Arceo, A.; Bennett, A.; Tham, M.; Olanrewaju, B.; McGrail, M.; Isin, K.; Olson, A. W.; Saxe, S. A Construction Classification System Database for Understanding Resource Use in Building Construction. Sci Data 2022, 9 (1). https://doi.org/10.1038/s41597-022-01141-8.
