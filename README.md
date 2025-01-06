# SMPL-Anthropometry

Measure the SMPL/SMPLX body models and visualize the measurements and landmarks.

<p align="center">
  <img src="https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/assets/measurement_visualization.png" width="950">
</p>

<br>

## 🔨 Getting started
You can use a docker container to facilitate running the code. Run in terminal:

```bash
cd docker
sh build.sh
sh docker_run.sh CODE_PATH
```

by adjusting the `CODE_PATH` to the `SMPL-Anthropometry` directory location. This creates a `smpl-anthropometry-container` container.

If you do not want to use a docker container, you can also just install the necessary packages from `docker/requirements.txt` into your own enviroment.

Next, provide the body models (SMPL or SMPLX) and:
1. put the `SMPL_{GENDER}.pkl` (MALE, FEMALE and NEUTRAL) models into the `data/smpl` folder
2. put the `SMPLX_{GENDER}.pkl` (MALE, FEMALE and NEUTRAL) models into the `data/smplx` folder

All the models can be found [here](https://github.com/vchoutas/smplx#downloading-the-model).

<br>

## 🏃 Running

First import the necessary libraries:

```python
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
```
<br>

Next define the measurer by setting the body model you want to measure with `model_type` (`smpl` or `smplx`):
```python
measurer = MeasureBody(model_type)
```
<br>

Then, there are two ways of using the code for measuring a body model depending on how you want to define the body:

1. Define the body model using the shape `betas` and gender `gender` parameters:

```python
measurer.from_body_model(gender=gender, shape=betas) 
```

2. Define the body model using the N x 3 vertices `verts` (N=6890 if SMPL, and 10475 if SMPLX):

```python
measurer.from_verts(verts=verts) 
```
&nbsp;&nbsp;&nbsp;&nbsp; 📣 Defining the body using the vertices can be especially useful when the SMPL/SMPLX vertices have been <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; further refined to fit a 2D/3D model and do not satsify perfectly a set of shape parameters anymore.<br>
<br>

Finally, you can measure the body with:
```python

measurement_names = measurer.all_possible_measurements # or chose subset of measurements 
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

where `measurer1` and `measurer2` are two intances of the `MeasureBody` class.

<br>

## 💿 Demos

You can run the `measure.py` script to measure all the predefined measurements (mentioned above) and visualize the results for a zero-shaped T-posed neutral gender SMPL body model:

```bash
python measure.py --measure_neutral_smpl_with_mean_shape
```

The output consists of a dictionary of measurements expressed in cm, the labeled measurements using standard labels,and the viualization of the measurements in the browser, as in the Figure above.

Similarly, you can measure a zero-shaped T-posed neutral gender SMPLX body model with:
```bash
python measure.py --measure_neutral_smplx_with_mean_shape
```

<br>

You can run the `evaluate.py` script to compare two sets of measurements of randomly shaped SMPL bodies as:

```python
python evaluate.py
```
The output consists of the mean absolute error (MAE) between two sets of measurements.

<br>
<br>

## 📝 Notes

### Measurement definitions
There are two types of measurements: lenghts and circumferences.
1. Lengths are defined as distances between landmark points defined on the body model
2. Circumferences are defiend as plane cuts of the body model

To define a new measurement:
1. Open `measurement_definitions.py`
1. add the new measurement to the `MEASUREMENT_TYPES` dict and set its type:
   `LENGTH` or `CIRCUMFERENCE`
2. depending on the measurement type, define the measurement in the `LENGTHS` or 
   `CIRCUMFERENCES` dict of the appropriate body model (`SMPLMeasurementDefinitions` or `SMPLXMeasurementDefinitions`)
   - `LENGTHS` are defined using 2 landmarks - the measurement is 
            found as the distance between the landmarks
   - `CIRCUMFERENCES` are defined with landmarks and joints - the 
            measurement is found by cutting the body model with the 
            plane defined by a point (landmark point) and normal (
            vector connecting the two joints)
3. If the measurement is a `CIRCUMFERENCE`, a possible issue that arises is
   that the plane cutting results in multiple body part slices. To alleviate
   that, define the body part where the measurement should be located in 
   `CIRCUMFERENCE_TO_BODYPARTS` dict. This way, only the slice in the corresponding body part is
   used for finding the measurement. The body parts are defined by the 
   face segmentation located in `data/smpl_body_parts_2_faces.json` or `data/smplx_body_parts_2_faces.json`.

<br>

### Measurement normalization
If a body model has unknown scale (ex. the body was regressed from an image), the measurements can be height-normalized as so:

```python
measurer = MeasureBody(model_type) # assume given model type
measurer.from_body_model(shape=betas, gender=gender) # assume given betas and gender

all_measurement_names = measurer.possible_measurements
measurer.measure(all_measurement_names)
new_height = 175
measurer.height_normalize_measurements(new_height)
```

This creates a dict of measurements `measurer.height_normalized_measurements` where each measurement was normalized with:
```
new_measurement = (old_measurement / old_height) * new_height
```
<br>

### Additional visualizations
To visualize the SMPL and SMPLX face segmentation on two separate plots, run:
```bash
python visualize.py --visualize_smpl_and_smplx_face_segmentation
```

To visualize the SMPL and SMPLX joints on the same plot, run:
```bash
python visualize.py --visualize_smpl_and_smplx_joints
```

To visualize the SMPL and SMPLX point segmentations on two side-by-side plots, run:
```bash
python visualize.py --visualize_smpl_and_smplx_point_segmentation
```
NOTE: You need to provide the `point_segmentation_meshcapade.json` files in the folders `data/smpl` and `data/smplx` from [here](https://meshcapade.wiki/SMPL#body-part-segmentation).

To visualize the SMPL and SMPLX landmarks on two side-by-side plots, run:
```bash
python visualize.py --visualize_smpl_and_smplx_landmarks
```

<br>
<br>

## 🤸‍♂️ Measuring posed subjects

This repository is meant to measure the SMPL family of models in the neutral T-pose. If you wish to measure a posed SMPL or scan of a subject, please refer to our [pose-independent anthropometry](https://github.com/DavidBoja/pose-independent-anthropometry/) repository.


<br>
<br>

## 🗞️ Citation

Please cite our work and leave a star ⭐ if you find the repository useful.

```bibtex
@misc{SMPL-Anthropometry,
  author = {Bojani\'{c}, D.},
  title = {SMPL-Anthropometry},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DavidBoja/SMPL-Anthropometry}},
}
```

<br>
<br>

## TODO

- [X] Implement SMPL-X body model
- [ ] Implement STAR body model
- [ ] Implement SUPR body model
- [X] Add height normalization for the measurements
- [ ] Allow posed and shaped body models as inputs, and measure them after unposing

<br>

⭐ <b>Leave a star if you find this repository useful</b> ⭐
