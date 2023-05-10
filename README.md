# SMPL-Anthropometry

Measure the SMPL body model and visualize the measurements and landmarks.

<p align="center">
  <img src="https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/assets/measurement_visualization.png" width="950">
</p>

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

## Running
You can use the demo script to measure all the predefined measurements and visualize the results.

```python
python demo.py
```

The output consists of a dictionary of measurements expressed in cm, the labeled measurements using standard labels, 
and the viualization of the measurements in the browser, as in the Figure above.



The list of the predefined measurements along with its standard literature labels are:

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

## NOTES

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


### Body model 
The body model can be defined by:
- the SMPL shape parameters β using `MeasureSMPL(...).from_smpl`
- the 6890 SMPL vertices, without the shape parameters, using `MeasureSMPL(...).from_verts`


## TODO

- [ ] Implement other body models (SMPL-X, STAR, ...)
- [ ] Add height normalization for the measurements
- [ ] Allow posed and shaped body models as inputs, and measure them after unposing


