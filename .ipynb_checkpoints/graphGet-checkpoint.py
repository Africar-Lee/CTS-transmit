import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import geopandas as gpd
import transbigdata as tbd
line, stop = tbd.getbusdata('北京', ['1号线'])

line.plot()
