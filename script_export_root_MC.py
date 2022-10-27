# TODO: change location to utilities

import numpy as np
import os

from sificc_lib.root_files import root_files
from sificc_lib.Preprocessing import Preprocessing

#######################################################
# scan for correct directory format

# get current directory
dir_main = os.getcwd()
# scan for data subdirectory, if not there crete it
try:
    os.chdir(dir_main + "/data/")
except:
    print("created directory: ", dir_main + "/data/")
    os.mkdir(dir_main + "/data/")
# change back to main directory
os.chdir(dir_main)

#######################################################
# global parameter

root_filename = dir_main + root_files.optimized_0mm_local

# filename of exported .npz file
npz_filename = "optimized_0mm_MC_test.npz"

#########################################################
# Define Monte-Carlo truths to be extracted

# define dataframe header
df_header = ["EventNumber",
             "MCSimulatedEventType",
             "BCIdentified",
             "NNIdentified",
             "NNPLACEHOLDER",  # for later use if needed
             "MCEnergy_e",
             "MCEnergy_p",
             "MCPosition_source.x",
             "MCPosition_source.y",
             "MCPosition_source.z",
             "MCDirection_source.x",
             "MCDirection_source.y",
             "MCDirection_source.z",
             "MCComptenPosition.x",
             "MCComptenPosition.y",
             "MCComptenPosition.z",
             "MCDirection_Scatter.x",
             "MCDirection_Scatter.y",
             "MCDirection_Scatter.z",
             "MCPosition_e.x",
             "MCPosition_e.y",
             "MCPosition_e.z",
             "MCPosition_p.x",
             "MCPosition_p.y",
             "MCPosition_p.z"]

###########################################################
# extract root-file

# create RootData object
root_data = Preprocessing(root_filename)

# create dataframe
df = np.zeros(shape=(root_data.num_entries, len(df_header)))

for i, event in enumerate(root_data.iterate_events(n=10000)):
    # define empty event data
    df_row = np.zeros(shape=(1, len(df_header)))
    # grab event data
    df_row[0, :] = [event.EventNumber,
                    event.MCSimulatedEventType,
                    event.Identified,
                    0.0,
                    0.0,
                    event.MCEnergy_e,
                    event.MCEnergy_p,
                    event.MCPosition_source.x,
                    event.MCPosition_source.y,
                    event.MCPosition_source.z,
                    event.MCDirection_source.x,
                    event.MCDirection_source.y,
                    event.MCDirection_source.z,
                    event.MCComptonPosition.x,
                    event.MCComptonPosition.y,
                    event.MCComptonPosition.z,
                    event.MCDirection_scatter.x,
                    event.MCDirection_scatter.y,
                    event.MCDirection_scatter.z,
                    event.MCPosition_e_first.x,
                    event.MCPosition_e_first.y,
                    event.MCPosition_e_first.z,
                    event.MCPosition_p_first.x,
                    event.MCPosition_p_first.y,
                    event.MCPosition_p_first.z]

    # write event data into dataframe
    df[i, :] = df_row

############################################################
# export dataframe to compressed .npz

with open(dir_main + "/data/" + npz_filename, 'wb') as file:
    np.savez_compressed(file, MC_TRUTH=df)

print("file saved as ", npz_filename)
