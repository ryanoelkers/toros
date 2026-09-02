import pandas as pd
from config import Configuration

vars = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_varstats.txt", sep=' ')
vars.to_csv('filtergraph.csv', na_rep='NULL', index=False, sep=',')

print('hold')