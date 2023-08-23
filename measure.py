
from typing import List, Dict
import numpy as np
import trimesh
import torch
import smplx
from pprint import pprint
import os
from measurement_definitions import *
from utils import *
from visualize import Visualizer


def set_shape(model, shape_coefs):
    '''
    Set shape of SMPL model.
    :param model: SMPL body model
    :param shape_coefs: torch.tensor dim (10,)

    Return
    shaped SMPL body model
    '''
    shape_coefs = shape_coefs.to(torch.float32)
    return model(betas=shape_coefs, return_verts=True)

def create_model(smpl_path, gender, num_coefs=10):
    '''
    Create SMPL body model
    :param smpl_path: str of location to SMPL .pkl models
    :param gender: str of gender: MALE or FEMALE or NEUTRAL
    :param num_coefs: int of number of SMPL shape coefficients
                      requires the model with num_coefs in smpl_path
    
    Return:
    :param SMPL body model
    '''
    
    body_pose = torch.zeros((1, (SMPL_NUM_KPTS-1) * 3))
    if smpl_path.split("/")[-1] == "smpl":
        smpl_path = os.path.dirname(smpl_path)
    
    return smplx.create(smpl_path, 
                        model_type="smpl",
                        gender=gender, 
                        use_face_contour=False,
                        num_betas=num_coefs,
                        body_pose=body_pose,
                        ext='pkl')

def get_SMPL_joint_regressor(smpl_path):
    '''
    Extract joint regressor from SMPL body model
    :param smpl_path: str of location to SMPL .pkl models
    
    Return:
    :param model.J_regressor: torch.tensor (23,6890) used to 
                              multiply with body model to get 
                              joint locations
    '''

    if smpl_path.split("/")[-1] == "smpl":
        smpl_path = os.path.dirname(smpl_path)

    model = smplx.create(smpl_path, 
                    model_type="smpl",
                    gender="MALE", 
                    use_face_contour=False,
                    num_betas=10,
                    body_pose=torch.zeros((1, 23 * 3)),
                    ext='pkl')
    return model.J_regressor


class MeasureSMPL():
    '''
    Measure the SMPL model defined either by the shape parameters or
    by its 6890 vertices. 

    All the measurements are expressed in cm.
    '''

    def __init__(self,
                 smpl_path: str
                ):
        self.smpl_path = smpl_path

        self.verts = None
        self.faces = smplx.SMPL(os.path.join(self.smpl_path,"smpl")).faces
        self.joints = None
        self.gender = None

        self.face_segmentation = load_face_segmentation(self.smpl_path)

        self.measurements = {}
        self.height_normalized_measurements = {}
        self.labeled_measurements = {}
        self.height_normalized_labeled_measurements = {}
        self.labels2names = {}
        self.landmarks = LANDMARK_INDICES
        self.measurement_types = MeasurementDefinitions().measurement_types
        self.length_definitions = MeasurementDefinitions().LENGTHS
        self.circumf_definitions = MeasurementDefinitions().CIRCUMFERENCES
        self.circumf_2_bodypart = MeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        self.cached_visualizations = {"LENGTHS":{}, "CIRCUMFERENCES":{}}
        self.all_possible_measurements = MeasurementDefinitions().possible_measurements

        # FIXME: this needs to be defined in init depending on model
        self.joint2ind = JOINT2IND

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (6890,3) of SMPL vertices
        '''        

        assert verts.shape == torch.Size([6890,3]), "verts need to be of dimension (6890,3)"

        joint_regressor = get_SMPL_joint_regressor(self.smpl_path)
        joints = torch.matmul(joint_regressor, verts)
        self.joints = joints.numpy()
        self.verts = verts.numpy()

    def from_smpl(self,
                  gender: str,
                  shape: torch.tensor):
        '''
        Construct body model from given gender and shape params 
        of SMPl model.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
                                    for SMPL model
        '''  
        model = create_model(self.smpl_path,gender)    
        model_output = set_shape(model, shape)
        
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender

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
        circumf_n1, circumf_n2 = JOINT2IND[circumf_n1], JOINT2IND[circumf_n2]
        
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



if __name__ == "__main__":

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