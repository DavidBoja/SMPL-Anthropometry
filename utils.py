

import os.path as osp
import json
import sys

def load_face_segmentation(smpl_path: str):
        '''
        Load smpl face segmentation which defines for each SMPL body part
        the faces that belong to it.
        :param smpl_path: str - path to SMPL files, including the 
                            smpl_body_parts_faces_meshcapade_labels.json file
        '''

        segmentation_path = osp.join(
                smpl_path,
                "smpl_body_parts_2_faces.json"
                )

        try:
            with open(segmentation_path, 'r') as f:
                face_segmentation = json.load(f)
        except FileNotFoundError:
            sys.exit(f"No such file - {segmentation_path}")

        return face_segmentation