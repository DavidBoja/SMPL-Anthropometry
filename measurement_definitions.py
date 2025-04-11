
from landmark_definitions import *
from joint_definitions import *

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


MEASUREMENT_TYPES = {
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
        "arm left length":  MeasurementType.LENGTH,
        "inside leg height": MeasurementType.LENGTH,
        "thigh left circumference": MeasurementType.CIRCUMFERENCE,
        "calf left circumference": MeasurementType.CIRCUMFERENCE,
        "ankle left circumference": MeasurementType.CIRCUMFERENCE,
        "shoulder breadth": MeasurementType.LENGTH,

        "arm length (shoulder to elbow)": MeasurementType.LENGTH,
        "arm length (spine to wrist)": MeasurementType.LENGTH,
        "crotch height": MeasurementType.LENGTH,
        "Hip circumference max height": MeasurementType.LENGTH
    }

class SMPLMeasurementDefinitions():
    '''
    Definition of SMPL measurements.

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
    
    LENGTHS = {"height": 
                    (SMPL_LANDMARK_INDICES["HEAD_TOP"], 
                     SMPL_LANDMARK_INDICES["HEELS"]
                     ),
               "shoulder to crotch height": 
                    (SMPL_LANDMARK_INDICES["SHOULDER_TOP"], 
                     SMPL_LANDMARK_INDICES["INSEAM_POINT"]
                    ),
                "arm left length": 
                    (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"], 
                     SMPL_LANDMARK_INDICES["LEFT_WRIST"]
                    ),
                "arm right length":
                    (SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"], 
                     SMPL_LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
                "inside leg height": 
                    (SMPL_LANDMARK_INDICES["LOW_LEFT_HIP"], 
                     SMPL_LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
                "shoulder breadth": 
                    (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"], 
                     SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
                "arm length (shoulder to elbow)":
                    (
                    #  SMPL_LANDMARK_INDICES["LEFT_SHOULDER"], 
                    #  SMPL_LANDMARK_INDICES["LEFT_ELBOW"]
                    SMPL_LANDMARK_INDICES["Rt. Acromion"],
                    SMPL_LANDMARK_INDICES["Rt. Humeral Lateral Epicn"]
                    ),
                "crotch height":
                    (SMPL_LANDMARK_INDICES["CROTCH"],
                     SMPL_LANDMARK_INDICES["HEELS"]
                    ),
                "Hip circumference max height":
                    (SMPL_LANDMARK_INDICES["PUBIC_BONE"],
                     SMPL_LANDMARK_INDICES["HEELS"]
                    ),
                # FIXME: implement geodesic distance for this measurement
                "arm length (spine to wrist)": 
                    (
                    #  SMPL_LANDMARK_INDICES["SHOULDER_TOP"], 
                    #  SMPL_LANDMARK_INDICES["LEFT_WRIST"]
                        SMPL_LANDMARK_INDICES["Cervicale"],
                        SMPL_LANDMARK_INDICES["Rt. Acromion"],
                        SMPL_LANDMARK_INDICES["Rt. Humeral Lateral Epicn"],
                        SMPL_LANDMARK_INDICES["Rt. Ulnar Styloid"]
                    ),
               }

    # defined with landmarks and joints
    # landmarks are defined with indices of the smpl model points
    # normals are defined with joint names of the smpl model
    CIRCUMFERENCES = {
        "head circumference":{"LANDMARKS":["HEAD_LEFT_TEMPLE"],
                               "JOINTS":["pelvis","spine3"]},

        "neck circumference":{"LANDMARKS":["NECK_ADAM_APPLE"],
                               "JOINTS":["spine2","head"]},
        
        "chest circumference":{"LANDMARKS":["LEFT_NIPPLE","RIGHT_NIPPLE"],
                               "JOINTS":["pelvis","spine3"]},

        "waist circumference":{"LANDMARKS":["BELLY_BUTTON","BACK_BELLY_BUTTON"],
                               "JOINTS":["pelvis","spine3"]},
        
        "hip circumference":{"LANDMARKS":["PUBIC_BONE"],
                               "JOINTS":["pelvis","spine3"]},
        
        "wrist right circumference":{"LANDMARKS":["RIGHT_WRIST"],
                                    "JOINTS":["right_wrist","right_hand"]},
        
        "bicep right circumference":{"LANDMARKS":["RIGHT_BICEP"],
                                    "JOINTS":["right_shoulder","right_elbow"]},

        "forearm right circumference":{"LANDMARKS":["RIGHT_FOREARM"],
                                        "JOINTS":["right_elbow","right_wrist"]},
        
        "thigh left circumference":{"LANDMARKS":["LEFT_THIGH"],
                                    "JOINTS":["pelvis","spine3"]},
        
        "calf left circumference":{"LANDMARKS":["LEFT_CALF"],
                                    "JOINTS":["pelvis","spine3"]},

        "ankle left circumference":{"LANDMARKS":["LEFT_ANKLE"],
                                    "JOINTS":["pelvis","spine3"]},      
                    
                    }
    
    possible_measurements = list(LENGTHS.keys()) + list(CIRCUMFERENCES.keys())

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



class SMPLXMeasurementDefinitions():
    '''
    Definition of SMPLX measurements.

    To add a new measurement:
    1. add it to the measurement_types dict and set the type:
       LENGTH or CIRCUMFERENCE
    2. depending on the type, define the measurement in LENGTHS or 
       CIRCUMFERENCES dict
       - LENGTHS are defined using 2 landmarks - the measurement is 
                found with distance between landmarks
       - CIRCUMFERENCES are defined with landmarks and joints - the 
                measurement is found by cutting the SMPLX model with the 
                plane defined by a point (landmark point) and normal (
                vector connecting the two joints)
    3. If the body part is a CIRCUMFERENCE, a possible issue that arises is
       that the plane cutting results in multiple body part slices. To alleviate
       that, define the body part where the measurement should be located in 
       CIRCUMFERENCE_TO_BODYPARTS dict. This way, only slice in that body part is
       used for finding the measurement. The body parts are defined by the SMPL 
       face segmentation.
    '''
    
    LENGTHS = {"height": 
                    (SMPLX_LANDMARK_INDICES["HEAD_TOP"], 
                     SMPLX_LANDMARK_INDICES["HEELS"]
                     ),
               "shoulder to crotch height": 
                    (SMPLX_LANDMARK_INDICES["SHOULDER_TOP"], 
                     SMPLX_LANDMARK_INDICES["INSEAM_POINT"]
                    ),
                "arm left length": 
                    (SMPLX_LANDMARK_INDICES["LEFT_SHOULDER"], 
                     SMPLX_LANDMARK_INDICES["LEFT_WRIST"]
                    ),
                "arm right length":
                    (SMPLX_LANDMARK_INDICES["RIGHT_SHOULDER"], 
                     SMPLX_LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
                "inside leg height": 
                    (SMPLX_LANDMARK_INDICES["LOW_LEFT_HIP"], 
                     SMPLX_LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
                "shoulder breadth": 
                    (SMPLX_LANDMARK_INDICES["LEFT_SHOULDER"], 
                     SMPLX_LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
               }

    # defined with landmarks and joints
    # landmarks are defined with indices of the smpl model points
    # normals are defined with joint names of the smpl model
    CIRCUMFERENCES = {
        "head circumference":{"LANDMARKS":["HEAD_LEFT_TEMPLE"],
                               "JOINTS":["pelvis","spine3"]},

        "neck circumference":{"LANDMARKS":["NECK_ADAM_APPLE"],
                               "JOINTS":["spine1","spine3"]},
        
        "chest circumference":{"LANDMARKS":["LEFT_NIPPLE","RIGHT_NIPPLE"],
                               "JOINTS":["pelvis","spine3"]},

        "waist circumference":{"LANDMARKS":["BELLY_BUTTON","BACK_BELLY_BUTTON"],
                               "JOINTS":["pelvis","spine3"]},
        
        "hip circumference":{"LANDMARKS":["PUBIC_BONE"],
                               "JOINTS":["pelvis","spine3"]},
        
        "wrist right circumference":{"LANDMARKS":["RIGHT_WRIST"],
                                    "JOINTS":["right_wrist","right_elbow"]}, # different from SMPL
        
        "bicep right circumference":{"LANDMARKS":["RIGHT_BICEP"],
                                    "JOINTS":["right_shoulder","right_elbow"]},

        "forearm right circumference":{"LANDMARKS":["RIGHT_FOREARM"],
                                        "JOINTS":["right_elbow","right_wrist"]},
        
        "thigh left circumference":{"LANDMARKS":["LEFT_THIGH"],
                                    "JOINTS":["pelvis","spine3"]},
        
        "calf left circumference":{"LANDMARKS":["LEFT_CALF"],
                                    "JOINTS":["pelvis","spine3"]},

        "ankle left circumference":{"LANDMARKS":["LEFT_ANKLE"],
                                    "JOINTS":["pelvis","spine3"]},      
                    
                    }
    
    possible_measurements = list(LENGTHS.keys()) + list(CIRCUMFERENCES.keys())

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