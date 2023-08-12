"""
Append location information to USV data
"""

import utils
from scipy.io import loadmat
import pandas as pd

# File locations
data_dir = '//Singingmouse/data/usv_calls/usv_note_analysis/03_div_cage_group01_18_song/'
usv_detections_filepath = data_dir + 'song detection files/'
location_data_filepath = data_dir + 'location data/'

location_filename = '2022-12-08-125822.common_coords'
detections_filename = 'T221208125757_0000111_030'

# Import locations file
data = loadmat(location_data_filepath+location_filename+'.mat')
l_locations = data['l_locations_world']
r_locations = data['r_locations_world']

# Import USV detections file
USV_detections = pd.read_csv(usv_detections_filepath+detections_filename+'.csv')

USV_detections = USV_detections.apply(utils.append_loc,axis=1,l_locations=l_locations,r_locations=r_locations)

# Save
USV_detections.to_csv(usv_detections_filepath+detections_filename+'_locations'+'.csv', index=False)