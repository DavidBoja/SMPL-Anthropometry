
import torch
from pprint import pprint

from measure import *
from measurement_definitions import *
from utils import *


smpl_path = "/SMPL-Anthropometry/data/SMPL"
measurer = MeasureSMPL(smpl_path=smpl_path)

betas = torch.zeros((1, 10), dtype=torch.float32)
measurer.from_smpl(gender="MALE", shape=betas)

measurement_names = MeasurementDefinitions.possible_measurements
measurer.measure(measurement_names)
print("Measurements")
pprint(measurer.measurements)

measurer.label_measurements(STANDARD_LABELS)
print("Labeled measurements")
pprint(measurer.labeled_measurements)

measurer.visualize()