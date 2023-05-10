

SMPL_NUM_KPTS = 24

# https://meshcapade.wiki/SMPL
IND2JOINT = {
    0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
}

STANDARD_LABELS = {
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


class MeasurementType():
    CIRCUMFERENCE = "circumference"
    LENGTH = "length"

class MeasurementDefinitions():
    '''
    Definition of SMPL landmarks and measurements.

    To add a new measurement:
    1. add it to the measurement_types dict and set the type:
       LENGTH or CIRCUMFERENCE
    2. depending on the type, define the measurement in LENGTHS or 
       CIRCUMFERENCES dict
       - LENGTHS are defined using 2 landmarks - the measurement is 
                found with distance between landmarks
       - CIRCUMFERENCES are defined with landmarks and joints - the 
                measurement is found by cutting the SMPL model with the 
                plane defined by a point (landmark point) and normal (
                vector connecting the two joints)
    3. If the body part is a CIRCUMFERENCE, a possible issue that arises is
       that the plane cutting results in multiple body part slices. To alleviate
       that, define the body part where the measurement should be located in 
       CIRCUMFERENCE_TO_BODYPARTS dict. This way, only slice in that body part is
       used for finding the measurement. The body parts are defined by the SMPL 
       face segmentation.
    '''

    measurement_types = {
        "height": MeasurementType.LENGTH,
        "head circumference": MeasurementType.CIRCUMFERENCE,
        "neck circumference": MeasurementType.CIRCUMFERENCE,
        "shoulder to crotch height": MeasurementType.LENGTH,
        "chest circumference": MeasurementType.CIRCUMFERENCE,
        "waist circumference": MeasurementType.CIRCUMFERENCE,
        "hip circumference": MeasurementType.CIRCUMFERENCE,

        "wrist right circumference": MeasurementType.CIRCUMFERENCE,
        "bicep right circumference": MeasurementType.CIRCUMFERENCE,
        "forearm right circumference": MeasurementType.CIRCUMFERENCE,
        "arm right length": MeasurementType.LENGTH,
        "inside leg height": MeasurementType.LENGTH,
        "thigh left circumference": MeasurementType.CIRCUMFERENCE,
        "calf left circumference": MeasurementType.CIRCUMFERENCE,
        "ankle left circumference": MeasurementType.CIRCUMFERENCE,
        "shoulder breadth": MeasurementType.LENGTH,
    }

    possible_measurements = measurement_types.keys()

    # Landmarks
    LANDMARK_INDICES = {"HEAD_TOP": 412,
                        "HEAD_LEFT_TEMPLE": 166,
                        "NECK_ADAM_APPLE": 3050,
                        "LEFT_HEEL": 3458,
                        "RIGHT_HEEL": 6858,
                        "LEFT_NIPPLE": 3042,
                        "RIGHT_NIPPLE": 6489,

                        "SHOULDER_TOP": 3068,
                        "INSEAM_POINT": 3149,
                        "BELLY_BUTTON": 3501,
                        "BACK_BELLY_BUTTON": 3022,
                        "CROTCH": 1210,
                        "PUBIC_BONE": 3145,
                        "RIGHT_WRIST": 5559,
                        "LEFT_WRIST": 2241,
                        "RIGHT_BICEP": 4855,
                        "RIGHT_FOREARM": 5197,
                        "LEFT_SHOULDER": 3011,
                        "RIGHT_SHOULDER": 6470,
                        "LEFT_ANKLE": 3334,
                        "LOW_LEFT_HIP": 3134,
                        "LEFT_THIGH": 947,
                        "LEFT_CALF": 1074,
                        "LEFT_ANKLE": 3325
                        }
    
    LANDMARK_INDICES["HEELS"] = (LANDMARK_INDICES["LEFT_HEEL"], 
                                 LANDMARK_INDICES["RIGHT_HEEL"])
    
    LENGTHS = {"height": 
                    (LANDMARK_INDICES["HEAD_TOP"], 
                     LANDMARK_INDICES["HEELS"]
                     ),
               "shoulder to crotch height": 
                    (LANDMARK_INDICES["SHOULDER_TOP"], 
                     LANDMARK_INDICES["INSEAM_POINT"]
                    ),
                "arm left length": 
                    (LANDMARK_INDICES["LEFT_SHOULDER"], 
                     LANDMARK_INDICES["LEFT_WRIST"]
                    ),
                "arm right length":
                    (LANDMARK_INDICES["RIGHT_SHOULDER"], 
                     LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
                "inside leg height": 
                    (LANDMARK_INDICES["LOW_LEFT_HIP"], 
                     LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
                "shoulder breadth": 
                    (LANDMARK_INDICES["LEFT_SHOULDER"], 
                     LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
               }

    # defined with landmarks and normals
    # landmarks are defined with the indices of the smpl model
    # normals are defined with joint indices of the smpl model
    CIRCUMFERENCES = {
        "head circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["HEAD_LEFT_TEMPLE"]],
                               "NORMAL_JOINTS_INDICES":[0,9]},

        "neck circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["NECK_ADAM_APPLE"]],
                               "NORMAL_JOINTS_INDICES":[12,15]},
        
        "chest circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["LEFT_NIPPLE"],
                                                   LANDMARK_INDICES["RIGHT_NIPPLE"]],
                               "NORMAL_JOINTS_INDICES":[0,9]},

        "waist circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["BELLY_BUTTON"],
                                                   LANDMARK_INDICES["BACK_BELLY_BUTTON"]],
                               "NORMAL_JOINTS_INDICES":[0,9]},
        
        "hip circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["PUBIC_BONE"]],
                               "NORMAL_JOINTS_INDICES":[0,9]},
        
        "wrist right circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["RIGHT_WRIST"]],
                               "NORMAL_JOINTS_INDICES":[21,23]},
        
        "bicep right circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["RIGHT_BICEP"]],
                               "NORMAL_JOINTS_INDICES":[17,19]},

        "forearm right circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["RIGHT_FOREARM"]],
                                        "NORMAL_JOINTS_INDICES":[19,21]},
        
        "thigh left circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["LEFT_THIGH"]],
                                        "NORMAL_JOINTS_INDICES":[0,9]},
        
        "calf left circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["LEFT_CALF"]],
                                        "NORMAL_JOINTS_INDICES":[0,9]},

        "ankle left circumference":{"LANDMARK_INDICES":[LANDMARK_INDICES["LEFT_ANKLE"]],
                                        "NORMAL_JOINTS_INDICES":[0,9]},      
                    
                    }

    CIRCUMFERENCE_TO_BODYPARTS = {
        "head circumference": "head",
        "neck circumference":"neck",
        "chest circumference":["spine1","spine2"],
        "waist circumference":["hips","spine"],
        "hip circumference":"hips",
        "wrist right circumference":["rightHand","rightForeArm"],
        "bicep right circumference":"rightArm",
        "forearm right circumference":"rightForeArm",
        "thigh left circumference": "leftUpLeg",
        "calf left circumference": "leftLeg",
        "ankle left circumference": "leftLeg",
    }