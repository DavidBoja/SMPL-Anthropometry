import smplx
import torch
import os

#from https://meshcapade.wiki/SMPL

SMPL_NUM_JOINTS = 24

SMPL_IND2JOINT = {
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

SMPL_JOINT2IND = {name:ind for ind,name in SMPL_IND2JOINT.items()}




SMPLX_NUM_JOINTS = 55

SMPLX_IND2JOINT = {
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
    22: 'jaw',
    23: 'left_eye',
    24: 'right_eye',
    25: 'left_index1',
    26: 'left_index2',
    27: 'left_index3',
    28: 'left_middle1',
    29: 'left_middle2',
    30: 'left_middle3',
    31: 'left_pinky1',
    32: 'left_pinky2',
    33: 'left_pinky3',
    34: 'left_ring1',
    35: 'left_ring2',
    36: 'left_ring3',
    37: 'left_thumb1',
    38: 'left_thumb2',
    39: 'left_thumb3',
    40: 'right_index1',
    41: 'right_index2',
    42: 'right_index3',
    43: 'right_middle1',
    44: 'right_middle2',
    45: 'right_middle3',
    46: 'right_pinky1',
    47: 'right_pinky2',
    48: 'right_pinky3',
    49: 'right_ring1',
    50: 'right_ring2',
    51: 'right_ring3',
    52: 'right_thumb1',
    53: 'right_thumb2',
    54: 'right_thumb3'
}

SMPLX_JOINT2IND = {name:ind for ind,name in SMPLX_IND2JOINT.items()}

def get_joint_regressor(body_model_type, body_model_root, gender="MALE", num_thetas=24):
    '''
    Extract joint regressor from SMPL body model
    :param body_model_type: str of body model type (smpl or smplx, etc.)
    :param body_model_root: str of location of folders where smpl/smplx 
                            inside which .pkl models 
    
    Return:
    :param model.J_regressor: torch.tensor (23,N) used to 
                              multiply with body model to get 
                              joint locations
    '''

    model = smplx.create(model_path=body_model_root, 
                        model_type=body_model_type,
                        gender=gender, 
                        use_face_contour=False,
                        num_betas=10,
                        body_pose=torch.zeros((1, num_thetas-1 * 3)),
                        ext='pkl')
    return model.J_regressor