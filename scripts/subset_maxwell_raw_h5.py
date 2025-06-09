# %%
from spikeinterface.extractors import MaxwellRecordingExtractor

# %%
print("Opening Maxwell raw h5 file...")
r = MaxwellRecordingExtractor("data/MEA_rec_patch_ground_truth_cell7.raw.h5")
