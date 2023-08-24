

import json
import sys
import numpy as np
from scipy.spatial import ConvexHull
import os
import argparse

def load_face_segmentation(path: str):
        '''
        Load face segmentation which defines for each body model part
        the faces that belong to it.
        :param path: str - path to json file with defined face segmentation
        '''

        try:
            with open(path, 'r') as f:
                face_segmentation = json.load(f)
        except FileNotFoundError:
            sys.exit(f"No such file - {path}")

        return face_segmentation


def convex_hull_from_3D_points(slice_segments: np.ndarray):
        '''
        Cretes convex hull from 3D points
        :param slice_segments: np.ndarray, dim N x 2 x 3 representing N 3D segments

        Returns:
        :param slice_segments_hull: np.ndarray, dim N x 2 x 3 representing N 3D segments
                                    that form the convex hull
        '''

        # stack all points in N x 3 array
        merged_segment_points = np.concatenate(slice_segments)
        unique_segment_points = np.unique(merged_segment_points,
                                            axis=0)

        # points lie in plane -- find which ax of x,y,z is redundant
        redundant_plane_coord = np.argmin(np.max(unique_segment_points,axis=0) - 
                                            np.min(unique_segment_points,axis=0) )
        non_redundant_coords = [x for x in range(3) if x!=redundant_plane_coord]

        # create convex hull
        hull = ConvexHull(unique_segment_points[:,non_redundant_coords])
        segment_point_hull_inds = hull.simplices.reshape(-1)

        slice_segments_hull = unique_segment_points[segment_point_hull_inds]
        slice_segments_hull = slice_segments_hull.reshape(-1,2,3)

        return slice_segments_hull


def filter_body_part_slices(slice_segments:np.ndarray, 
                             sliced_faces:np.ndarray,
                             measurement_name: str,
                             circumf_2_bodypart: dict,
                             face_segmentation: dict
                            ):
        '''
        Remove segments that are not in the appropriate body part 
        for the given measurement.
        :param slice_segments: np.ndarray - (N,2,3) for N segments 
                                            represented as two 3D points
        :param sliced_faces: np.ndarray - (N,) representing the indices of the
                                            faces
        :param measurement_name: str - name of the measurement
        :param circumf_2_bodypart: dict - dict mapping measurement to body part
        :param face_segmentation: dict - dict mapping body part to all faces belonging
                                        to it

        Return:
        :param slice_segments: np.ndarray (K,2,3) where K < N, for K segments 
                                represented as two 3D points that are in the 
                                appropriate body part
        '''

        if measurement_name in circumf_2_bodypart.keys():

            body_parts = circumf_2_bodypart[measurement_name]

            if isinstance(body_parts,list):
                body_part_faces = [face_index for body_part in body_parts 
                                    for face_index in face_segmentation[body_part]]
            else:
                body_part_faces = face_segmentation[body_parts]

            N_sliced_faces = sliced_faces.shape[0]

            keep_segments = []
            for i in range(N_sliced_faces):
                if sliced_faces[i] in body_part_faces:
                    keep_segments.append(i)

            return slice_segments[keep_segments]

        else:
            return slice_segments


def point_segmentation_to_face_segmentation(
                point_segmentation: dict,
                faces: np.ndarray,
                save_as: str = None):
    """
    :param point_segmentation: dict - dict mapping body part to 
                                      all points belonging to it
    :param faces: np.ndarray - (N,3) representing the indices of the faces
    :param save_as: str - optional path to save face segmentation as json
    """

    import json
    from tqdm import tqdm
    from collections import Counter

    # create body parts to index mapping
    mapping_bp2ind = dict(zip(point_segmentation.keys(),
                              range(len(point_segmentation.keys()))))
    mapping_ind2bp = {v:k for k,v in mapping_bp2ind.items()}


    # assign each face to body part index
    faces_segmentation = np.zeros_like(faces)
    for i,face in tqdm(enumerate(faces)):
        for bp_name, bp_indices in point_segmentation.items():
            bp_label = mapping_bp2ind[bp_name]
            
            for k in range(3):
                if face[k] in bp_indices:
                    faces_segmentation[i,k] = bp_label
    

    # for each face, assign the most common body part
    face_segmentation_final = np.zeros(faces_segmentation.shape[0])
    for i,f in enumerate(faces_segmentation):
        c = Counter(list(f))
        face_segmentation_final[i] = c.most_common()[0][0]
     

    # create dict with body part as key and faces as values
    face_segmentation_dict = {k:[] for k in mapping_bp2ind.keys()} 
    for i,fff in enumerate(face_segmentation_final):
        face_segmentation_dict[mapping_ind2bp[int(fff)]].append(i)


    # save face segmentation
    if save_as:
        with open(save_as, 'w') as f:
            json.dump(face_segmentation_dict, f)

    return face_segmentation_dict
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create face segmentation from \
                                     point segmentation of smpl/smplx models.')
    parser.add_argument('--create_face_segmentation', action='store_true')
    args = parser.parse_args()
    
    if args.create_face_segmentation:

        import smplx

        segm_path = "data/smplx/point_segmentation_meshcapade.json"
        with open(segm_path,"r") as f:
            point_segmentation = json.load(f)

        model_path = "data/smplx"
        smplx_faces = smplx.SMPLX(model_path,ext="pkl").faces

        save_as = "data/smplx/smplx_body_parts_2_faces.json"

        _ = point_segmentation_to_face_segmentation(point_segmentation,
                                                    smplx_faces,
                                                    save_as)