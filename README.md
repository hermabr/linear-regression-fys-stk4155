# FYS-STK4155 - Project 1

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements

```bash
pip install -r requirements.txt
```

## Structure

The repo is divided into two main parts:

-   `src` : This is where the python code and the data from the input and output is put
-   `report` : This is where the report and the latex is located

### Python code

```
.
├── config.py - config variables
├── data
│  └── geodata.tif
├── data_generation.py - used for generating franke data and for loading the data from the terrain data
├── lasso.py - for fitting and predicting using lasso regression
├── linear_regression_models.py - an abstract linear regression method housing many of the helper methods and the abstract structure for the other regression files
├── ordinary_least_squares.py - for fitting the ordinary least squares and for calculating confidence intervals
├── output - the output from run.py
│  ├── ...
├── plotter.py - a helper method for plotting different data
├── ridge.py - for fitting ridge regression and for calculating the confidence intervals
└── run.py - generating the plots and testing the code
```

## Usage

To run the "main" program which generates the plots, and for general code examples for using different methods, look at `run.py`

```bash
cd src
python run.py
```

The code is written to be pretty general, meaning that it should be easy to use some of the code in another project. If one for instance wanted to use the ordinary least squares method, it can be done using the following code:

```python
from data_generation import FrankeData
from plotter import surface_plot_raveled
from ordinary_least_squares import OrdinaryLeastSquares

data = FrankeData(30)

ols = OrdinaryLeastSquares(5)
ols.fit(data.x, data.y, data.z)

plotting_data = FrankeData(20)
z_tilde = ols.predict(plotting_data.x, plotting_data.y)

surface_plot_raveled(
    "Ordinary least squares",
    plotting_data.x,
    plotting_data.y,
    [[plotting_data.z, z_tilde]],
    [["z_test", "z-tilde"]],
    plotting_data.dimensions,
)
```
