
from typing import List, Dict
import numpy as np
import trimesh
import torch
import smplx
from pprint import pprint
import os
import argparse

from measurement_definitions import *
from utils import *
from visualize import Visualizer
from landmark_definitions import *
from joint_definitions import *



def set_shape(model, shape_coefs):
    '''
    Set shape of body model.
    :param model: smplx body model
    :param shape_coefs: torch.tensor dim (10,)

    Return
    shaped smplx body model
    '''
    shape_coefs = shape_coefs.to(torch.float32)
    return model(betas=shape_coefs, return_verts=True)

def create_model(model_type, model_root, gender, num_betas=10, num_thetas=24):
    '''
    Create SMPL/SMPLX/etc. body model
    :param model_type: str of model type: smpl, smplx, etc.
    :param model_root: str of location where there are smpl/smplx/etc. folders with .pkl models
                        (clumsy definition in smplx package)
    :param gender: str of gender: MALE or FEMALE or NEUTRAL
    :param num_betas: int of number of shape coefficients
                      requires the model with num_coefs in model_root
    :param num_thetas: int of number of pose coefficients
    
    Return:
    :param smplx body model (SMPL, SMPLX, etc.)
    '''
    
    #body_pose = torch.zeros((1, (num_thetas-1) * 3))
    
    return smplx.create(model_path=model_root,
                        model_type=model_type,
                        gender=gender, 
                        use_face_contour=False,
                        num_betas=num_betas,
                        #body_pose=body_pose,
                        ext='pkl')



class Measurer():
    '''
    Measure a parametric body model defined either.
    Parent class for Measure{SMPL,SMPLX,..}.

    All the measurements are expressed in cm.
    '''

    def __init__(self):
        self.verts = None
        self.faces = None
        self.joints = None
        self.gender = None

        self.measurements = {}
        self.height_normalized_measurements = {}
        self.labeled_measurements = {}
        self.height_normalized_labeled_measurements = {}
        self.labels2names = {}

    def from_verts(self):
        pass

    def from_body_model(self):
        pass

    def measure(self, 
                measurement_names: List[str]
                ):
        '''
        Measure the given measurement names from measurement_names list
        :param measurement_names - list of strings of defined measurements
                                    to measure from MeasurementDefinitions class
        '''

        for m_name in measurement_names:
            if m_name not in self.all_possible_measurements:
                print(f"Measurement {m_name} not defined.")
                pass

            if m_name in self.measurements:
                pass

            if self.measurement_types[m_name] == MeasurementType().LENGTH:

                value = self.measure_length(m_name)
                self.measurements[m_name] = value

            elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:

                value = self.measure_circumference(m_name)
                self.measurements[m_name] = value
    
            else:
                print(f"Measurement {m_name} not defined")

    def measure_length(self, measurement_name: str):
        '''
        Measure distance between 2 landmarks
        :param measurement_name: str - defined in MeasurementDefinitions

        Returns
        :float of measurement in cm
        '''

        measurement_landmarks_inds = self.length_definitions[measurement_name]

        landmark_points = []
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i],tuple):
                # if touple of indices for landmark, take their average
                lm = (self.verts[measurement_landmarks_inds[i][0]] + 
                          self.verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm = self.verts[measurement_landmarks_inds[i]]
            
            landmark_points.append(lm)

        landmark_points = np.vstack(landmark_points)[None,...]

        return self._get_dist(landmark_points)

    @staticmethod
    def _get_dist(verts: np.ndarray) -> float:
        '''
        The Euclidean distance between vertices.
        The distance is found as the sum of each pair i 
        of 3D vertices (i,0,:) and (i,1,:) 
        :param verts: np.ndarray (N,2,3) - vertices used 
                        to find distances
        
        Returns:
        :param dist: float, sumed distances between vertices
        '''

        verts_distances = np.linalg.norm(verts[:, 1] - verts[:, 0],axis=1)
        distance = np.sum(verts_distances)
        distance_cm = distance * 100 # convert to cm
        return distance_cm
    
    def measure_circumference(self, 
                              measurement_name: str, 
                              ):
        '''
        Measure circumferences. Circumferences are defined with 
        landmarks and joints - the measurement is found by cutting the 
        SMPL model with the  plane defined by a point (landmark point) and 
        normal (vector connecting the two joints).
        :param measurement_name: str - measurement name

        Return
        float of measurement value in cm
        '''

        measurement_definition = self.circumf_definitions[measurement_name]
        circumf_landmarks = measurement_definition["LANDMARKS"]
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = self.joint2ind[circumf_n1], self.joint2ind[circumf_n2]
        
        plane_origin = np.mean(self.verts[circumf_landmark_indices,:],axis=0)
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

        # new version            
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh, 
                                plane_normal=plane_normal, 
                                plane_origin=plane_origin, 
                                return_faces=True) # (N, 2, 3), (N,)
        
        slice_segments = filter_body_part_slices(slice_segments,
                                                 sliced_faces,
                                                 measurement_name,
                                                 self.circumf_2_bodypart,
                                                 self.face_segmentation)
        
        slice_segments_hull = convex_hull_from_3D_points(slice_segments)

        return self._get_dist(slice_segments_hull)

    def height_normalize_measurements(self, new_height: float):
        ''' 
        Scale all measurements so that the height measurement gets
        the value of new_height:
        new_measurement = (old_measurement / old_height) * new_height
        NOTE the measurements and body model remain unchanged, a new
        dictionary height_normalized_measurements is created.
        
        Input:
        :param new_height: float, the newly defined height.

        Return:
        self.height_normalized_measurements: dict of 
                {measurement:value} pairs with 
                height measurement = new_height, and other measurements
                scaled accordingly
        '''
        if self.measurements != {}:
            old_height = self.measurements["height"]
            for m_name, m_value in self.measurements.items():
                norm_value = (m_value / old_height) * new_height
                self.height_normalized_measurements[m_name] = norm_value

            if self.labeled_measurements != {}:
                for m_name, m_value in self.labeled_measurements.items():
                    norm_value = (m_value / old_height) * new_height
                    self.height_normalized_labeled_measurements[m_name] = norm_value

    def label_measurements(self,set_measurement_labels: Dict[str, str]):
        '''
        Create labeled_measurements dictionary with "label: x cm" structure
        for each given measurement.
        NOTE: This overwrites any prior labeling!
        
        :param set_measurement_labels: dict of labels and measurement names
                                        (example. {"A": "head_circumference"})
        '''

        if self.labeled_measurements != {}:
            print("Overwriting old labels")

        self.labeled_measurements = {}
        self.labels2names = {}

        for set_label, set_name in set_measurement_labels.items():
            
            if set_name not in self.all_possible_measurements:
                print(f"Measurement {set_name} not defined.")
                pass

            if set_name not in self.measurements.keys():
                self.measure([set_name])

            self.labeled_measurements[set_label] = self.measurements[set_name]
            self.labels2names[set_label] = set_name

    def visualize(self,
                 measurement_names: List[str] = [], 
                 landmark_names: List[str] = [],
                 title="Measurement visualization",
                 visualize_body: bool = True,
                 visualize_landmarks: bool = True,
                 visualize_joints: bool = True,
                 visualize_measurements: bool=True):

        # TODO: create default model if not defined
        # if self.verts is None:
        #     print("Model has not been defined. \
        #           Visualizing on default male model")
        #     model = create_model(self.smpl_path, "MALE", num_coefs=10)
        #     shape = torch.zeros((1, 10), dtype=torch.float32)
        #     model_output = set_shape(model, shape)
            
        #     verts = model_output.vertices.detach().cpu().numpy().squeeze()
        #     faces = model.faces.squeeze()
        # else:
        #     verts = self.verts
        #     faces = self.faces 

        if measurement_names == []:
            measurement_names = self.all_possible_measurements

        if landmark_names == []:
            landmark_names = list(self.landmarks.keys())

        vizz = Visualizer(verts=self.verts,
                        faces=self.faces,
                        joints=self.joints,
                        landmarks=self.landmarks,
                        measurements=self.measurements,
                        measurement_types=self.measurement_types,
                        length_definitions=self.length_definitions,
                        circumf_definitions=self.circumf_definitions,
                        joint2ind=self.joint2ind,
                        circumf_2_bodypart=self.circumf_2_bodypart,
                        face_segmentation=self.face_segmentation,
                        visualize_body=visualize_body,
                        visualize_landmarks=visualize_landmarks,
                        visualize_joints=visualize_joints,
                        visualize_measurements=visualize_measurements,
                        title=title
                        )
        
        vizz.visualize(measurement_names=measurement_names,
                       landmark_names=landmark_names,
                       title=title)


class MeasureSMPL(Measurer):
    '''
    Measure the SMPL model defined either by the shape parameters or
    by its 6890 vertices. 

    All the measurements are expressed in cm.
    '''

    def __init__(self):
        
        super().__init__()

        self.model_type = "smpl"
        self.body_model_root = "data"
        self.body_model_path = os.path.join(self.body_model_root, 
                                            self.model_type)

        self.faces = smplx.SMPL(self.body_model_path, ext="pkl").faces
        face_segmentation_path = os.path.join(self.body_model_path,
                                              f"{self.model_type}_body_parts_2_faces.json")
        self.face_segmentation = load_face_segmentation(face_segmentation_path)

        self.landmarks = SMPL_LANDMARK_INDICES
        self.measurement_types = MEASUREMENT_TYPES
        self.length_definitions = SMPLMeasurementDefinitions().LENGTHS
        self.circumf_definitions = SMPLMeasurementDefinitions().CIRCUMFERENCES
        self.circumf_2_bodypart = SMPLMeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        self.all_possible_measurements = SMPLMeasurementDefinitions().possible_measurements

        self.joint2ind = SMPL_JOINT2IND
        self.num_joints = SMPL_NUM_JOINTS

        self.num_points = 6890

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (6890,3) of SMPL vertices
        '''        

        verts = verts.squeeze()
        error_msg = f"verts need to be of dimension ({self.num_points},3)"
        assert verts.shape == torch.Size([self.num_points,3]), error_msg

        joint_regressor = get_joint_regressor(self.model_type, 
                                              self.body_model_root,
                                              gender="NEUTRAL", 
                                              num_thetas=self.num_joints)
        joints = torch.matmul(joint_regressor, verts)
        self.joints = joints.numpy()
        self.verts = verts.numpy()

    def from_body_model(self,
                        gender: str,
                        shape: torch.tensor):
        '''
        Construct body model from given gender and shape params 
        of SMPl model.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
                                    for SMPL model
        '''  

        model = create_model(model_type=self.model_type, 
                             model_root=self.body_model_root, 
                             gender=gender,
                             num_betas=10,
                             num_thetas=self.num_joints)    
        model_output = set_shape(model, shape)
        
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender


class MeasureSMPLX(Measurer):
    '''
    Measure the SMPLX model defined either by the shape parameters or
    by its 10475 vertices. 

    All the measurements are expressed in cm.
    '''

    def __init__(self):
        
        super().__init__()

        self.model_type = "smplx"
        self.body_model_root = "data"
        self.body_model_path = os.path.join(self.body_model_root, 
                                            self.model_type)

        self.faces = smplx.SMPLX(self.body_model_path, ext="pkl").faces
        face_segmentation_path = os.path.join(self.body_model_path,
                                              f"{self.model_type}_body_parts_2_faces.json")
        self.face_segmentation = load_face_segmentation(face_segmentation_path)

        self.landmarks = SMPLX_LANDMARK_INDICES
        self.measurement_types = MEASUREMENT_TYPES
        self.length_definitions = SMPLXMeasurementDefinitions().LENGTHS
        self.circumf_definitions = SMPLXMeasurementDefinitions().CIRCUMFERENCES
        self.circumf_2_bodypart = SMPLXMeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        self.all_possible_measurements = SMPLXMeasurementDefinitions().possible_measurements

        self.joint2ind = SMPLX_JOINT2IND
        self.num_joints = SMPLX_NUM_JOINTS

        self.num_points = 10475

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (10475,3) of SMPLX vertices
        '''        

        verts = verts.squeeze()
        error_msg = f"verts need to be of dimension ({self.num_points},3)"
        assert verts.shape == torch.Size([self.num_points,3]), error_msg

        joint_regressor = get_joint_regressor(self.model_type, 
                                              self.body_model_root,
                                              gender="NEUTRAL", 
                                              num_thetas=self.num_joints)
        joints = torch.matmul(joint_regressor, verts)
        self.joints = joints.numpy()
        self.verts = verts.numpy()

    def from_body_model(self,
                        gender: str,
                        shape: torch.tensor):
        '''
        Construct body model from given gender and shape params 
        of SMPl model.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
                                    for SMPL model
        '''  

        model = create_model(model_type=self.model_type, 
                             model_root=self.body_model_root, 
                             gender=gender,
                             num_betas=10,
                             num_thetas=self.num_joints)    
        model_output = set_shape(model, shape)
        
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender


class MeasureBody():
    def __new__(cls, model_type):
        model_type = model_type.lower()
        if model_type == 'smpl':
            return MeasureSMPL()
        elif model_type == 'smplx':
            return MeasureSMPLX()
        else:
            raise NotImplementedError("Model type not defined")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Measure body models.')
    parser.add_argument('--measure_neutral_smpl_with_mean_shape', action='store_true',
                        help="Measure a mean shape smpl model.")
    parser.add_argument('--measure_neutral_smplx_with_mean_shape', action='store_true',
                        help="Measure a mean shape smplx model.")
    args = parser.parse_args()

    model_types_to_measure = []
    if args.measure_neutral_smpl_with_mean_shape:
        model_types_to_measure.append("smpl")
    elif args.measure_neutral_smplx_with_mean_shape:
        model_types_to_measure.append("smplx")

    for model_type in model_types_to_measure:
        print(f"Measuring {model_type} body model")
        measurer = MeasureBody(model_type)

        betas = torch.zeros((1, 10), dtype=torch.float32)
        measurer.from_body_model(gender="NEUTRAL", shape=betas)

        measurement_names = measurer.all_possible_measurements
        measurer.measure(measurement_names)
        print("Measurements")
        pprint(measurer.measurements)

        measurer.label_measurements(STANDARD_LABELS)
        print("Labeled measurements")
        pprint(measurer.labeled_measurements)

        measurer.visualize()