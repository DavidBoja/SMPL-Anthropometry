# SMPL-Anthropometry

Measure the SMPL body model and visualize the measurements and landmarks.

<p align="center">
  <img src="https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/assets/measurement_visualization.png" width="950">
</p>

<br>

## Getting started
You can use a docker container to facilitate running the code. Run in terminal:

```bash
cd docker
sh build.sh
sh docker_run.sh CODE_PATH
```

by adjusting the `CODE_PATH` to the `SMPL-Anthropometry` directory location. This creates a `smpl-anthropometry-container` container.

If you do not want to use a docker container, you can also just install the necessary packages from `docker/requirements.txt` into your own enviroment.

Next, you need to provide the SMPL body models `SMPL_{GENDER}.pkl` (MALE, FEMALE and NEUTRAL), and put them into the `data/SMPL/smpl` folder.

<br>

## Running

First import the necessary libraries and define the SMPL bodies path:

```python
from measure import MeasureSMPL
from measurement_definitions import MeasurementDefinitions, STANDARD_LABELS

smpl_path = "/SMPL-Anthropometry/data/SMPL" 
```
<br>

Then, there are two ways of using the code for measuring an SMPL body model depending on how you want to define the body:

1. Define the SMPL body using the shape `betas` and gender `gender` parameters:

```python
measurer = MeasureSMPL(smpl_path=smpl_path) 
measurer.from_smpl(gender=gender, shape=betas) 
```

2. Define the SMPL body using the 6890 x 3 vertices `verts`:

```python
measurer = MeasureSMPL(smpl_path=smpl_path) 
measurer.from_verts(verts=verts) 
```
&nbsp;&nbsp;&nbsp;&nbsp; Defining the body using the vertices can be especially useful when the SMPL vertices have been further refined to fit a 2D/3D model <br>
&nbsp;&nbsp;&nbsp;&nbsp; and do not satsify perfectly a set of shape parameters anymore.

<br>

Finally, you can measure the SMPL bodies with:
```python
measurement_names = MeasurementDefinitions.possible_measurements 
measurer.measure(measurement_names) 
measurer.label_measurements(STANDARD_LABELS) 
```

Then, the measurements dictionary can be obtained with `measurer.measurements` and the labeled measurements can be obtained with `measurer.labeled_measurements`. The list of the predefined measurements along with its standard literature labels are:

```
STANDARD_MEASUREMENT = {
    'A': 'head circumference',
    'B': 'neck circumference',
    'C': 'shoulder to crotch height',
    'D': 'chest circumference',
    'E': 'waist circumference',
    'F': 'hip circumference',
    'G': 'wrist right circumference',
    'H': 'bicep right circumference',
    'I': 'forearm right circumference',
    'J': 'arm right length',
    'K': 'inside leg height',
    'L': 'thigh left circumference',
    'M': 'calf left circumference',
    'N': 'ankle left circumference',
    'O': 'shoulder breadth',
    'P': 'height'
    }
```

All the measurements are expressed in cm.

<br>

You can also compute the mean absolute error (MAE) between two sets of measurements as:
```python
from evaluate import evaluate_mae
MAE = evaluate_mae(measurer1.measurements,measurer2.measurements)
```

where `measurer1` and `measurer2` are two intances of the `MeasureSMPL` class.

<br>

## Demos

You can run the `measure.py` script to measure all the predefined measurements (mentioned above) and visualize the results for a zero-shaped T-posed SMPL body model:

```python
python measure.py
```

The output consists of a dictionary of measurements expressed in cm, the labeled measurements using standard labels,and the viualization of the measurements in the browser, as in the Figure above.


<br>

You can run the `evaluate.py` script to compare two sets of measurements of randomly shaped SMPL bodies as:

```python
python evaluate.py
```
The output consists of the mean absolute error (MAE) between two sets of measurements.

<br>
<br>

## Notes

### Measurement definitions
There are two types of measurements: lenghts and circumferences.
1. Lengths are defined as distances between landmark points defined on the SMPL body
2. Circumferences are defiend as plane cuts of the SMPL body

To define a new measurement:
1. Open `measurement_definitions.py`
1. add the new measurement to the `measurement_types` dict and set its type:
   `LENGTH` or `CIRCUMFERENCE`
2. depending on the type, define the measurement in the `LENGTHS` or 
   `CIRCUMFERENCES` dict
   - `LENGTHS` are defined using 2 landmarks - the measurement is 
            found as the distance between the landmarks
   - `CIRCUMFERENCES` are defined with landmarks and joints - the 
            measurement is found by cutting the SMPL model with the 
            plane defined by a point (landmark point) and normal (
            vector connecting the two joints)
3. If the body part is a `CIRCUMFERENCE`, a possible issue that arises is
   that the plane cutting results in multiple body part slices. To alleviate
   that, define the body part where the measurement should be located in 
   `CIRCUMFERENCE_TO_BODYPARTS` dict. This way, only the slice in the corresponding body part is
   used for finding the measurement. The body parts are defined by the SMPL 
   face segmentation located in `data/smpl_body_parts_2_faces.json`.

<br>

### Measurement normalization
If a body model has unknown scale (ex. the body was regressed from an image), the measurements can be height-normalized as so:

```python
measurer = MeasureSMPL()
measurer.from_smpl(shape=betas, gender=gender) # assume given betas and gender

all_measurement_names = MeasurementDefinitions.possible_measurements
measurer.measure(all_measurement_names)
new_height = 175
measurer.height_normalize_measurements(new_height)
```

This creates a dict of measurements `measurer.height_normalized_measurements` where each measurement was normalized with:
```
new_measurement = (old_measurement / old_height) * new_height
```

<br>
<br>

## TODO

- [ ] Implement other body models (SMPL-X, STAR, ...)
- [X] Add height normalization for the measurements
- [ ] Allow posed and shaped body models as inputs, and measure them after unposing


