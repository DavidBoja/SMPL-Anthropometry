
import numpy as np
from typing import List
import plotly
import plotly.graph_objects as go
import plotly.express as px
import trimesh
import argparse
import smplx
import json
import torch
#import ipywidgets as widgets
from plotly.subplots import make_subplots



from measurement_definitions import MeasurementType
from utils import convex_hull_from_3D_points, filter_body_part_slices
from joint_definitions import SMPL_IND2JOINT, SMPLX_IND2JOINT
from landmark_definitions import SMPL_LANDMARK_INDICES, SMPLX_LANDMARK_INDICES

class Visualizer():
    '''
    Visualize the body model with measurements, landmarks and joints.
    All the measurements are expressed in cm.
    '''

    def __init__(self,
                 verts: np.ndarray,
                 faces: np.ndarray,
                 joints: np.ndarray,
                 landmarks: dict,
                 measurements: dict,
                 measurement_types: dict,
                 length_definitions: dict,
                 circumf_definitions: dict,
                 joint2ind: dict,
                 circumf_2_bodypart: dict,
                 face_segmentation: dict,
                 visualize_body: bool = True,
                 visualize_landmarks: bool = True,
                 visualize_joints: bool = True,
                 visualize_measurements: bool=True,
                 title: str = "Measurement visualization"
                ):
        

        self.verts = verts
        self.faces = faces
        self.joints = joints
        self.landmarks = landmarks
        self.measurements = measurements
        self.measurement_types = measurement_types
        self.length_definitions = length_definitions
        self.circumf_definitions = circumf_definitions
        self.joint2ind = joint2ind
        self.circumf_2_bodypart = circumf_2_bodypart
        self.face_segmentation = face_segmentation

        self.visualize_body = visualize_body
        self.visualize_landmarks = visualize_landmarks
        self.visualize_joints = visualize_joints
        self.visualize_measurements = visualize_measurements

        self.title = title
        
      

    @staticmethod
    def create_mesh_plot(verts: np.ndarray, faces: np.ndarray):
        '''
        Visualize smpl body mesh.
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices

        Return:
        plotly Mesh3d object for plotting
        '''
        mesh_plot = go.Mesh3d(
                            x=verts[:,0],
                            y=verts[:,1],
                            z=verts[:,2],
                            color="gray",
                            hovertemplate ='<i>Index</i>: %{text}',
                            text = [i for i in range(verts.shape[0])],
                            # i, j and k give the vertices of triangles
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2],
                            opacity=0.6,
                            name='body',
                            )
        return mesh_plot
        
    @staticmethod
    def create_joint_plot(joints: np.ndarray):

        return go.Scatter3d(x = joints[:,0],
                            y = joints[:,1], 
                            z = joints[:,2], 
                            mode='markers',
                            marker=dict(size=8,
                                        color="black",
                                        opacity=1,
                                        symbol="cross"
                                        ),
                            name="joints"
                                )
    
    @staticmethod
    def create_wireframe_plot(verts: np.ndarray,faces: np.ndarray):
        '''
        Given vertices and faces, creates a wireframe of plotly segments.
        Used for visualizing the wireframe.
        
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the verts
        '''
        i=faces[:,0]
        j=faces[:,1]
        k=faces[:,2]

        triangles = np.vstack((i,j,k)).T

        x=verts[:,0]
        y=verts[:,1]
        z=verts[:,2]

        vertices = np.vstack((x,y,z)).T
        tri_points = vertices[triangles]

        #extract the lists of x, y, z coordinates of the triangle 
        # vertices and connect them by a "line" by adding None
        # this is a plotly convention for plotting segments
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])

        # return Xe, Ye, Ze 
        wireframe = go.Scatter3d(
                        x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        name='wireframe',
                        line=dict(color= 'rgb(70,70,70)', width=1)
                        )
        return wireframe

    def create_landmarks_plot(self,
                              landmark_names: List[str], 
                              verts: np.ndarray
                              ) -> List[plotly.graph_objs.Scatter3d]:
        '''
        Visualize landmarks from landmark_names list
        :param landmark_names: List[str] of landmark names to visualize

        Return
        :param plots: list of plotly objects to plot
        '''

        plots = []

        landmark_colors = dict(zip(self.landmarks.keys(),
                                px.colors.qualitative.Alphabet))

        for lm_name in landmark_names:
            if lm_name not in self.landmarks.keys():
                print(f"Landmark {lm_name} is not defined.")
                pass

            lm_index = self.landmarks[lm_name]
            if isinstance(lm_index,tuple):
                lm = (verts[lm_index[0]] + verts[lm_index[1]]) / 2
            else:
                lm = verts[lm_index] 

            plot = go.Scatter3d(x = [lm[0]],
                                y = [lm[1]], 
                                z = [lm[2]], 
                                mode='markers',
                                marker=dict(size=8,
                                            color=landmark_colors[lm_name],
                                            opacity=1,
                                            ),
                               name=lm_name
                                )

            plots.append(plot)

        return plots

    def create_measurement_length_plot(self, 
                                       measurement_name: str,
                                       verts: np.ndarray,
                                       color: str
                                       ):
        '''
        Create length measurement plot.
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        '''
        
        measurement_landmarks_inds = self.length_definitions[measurement_name]

        segments = {"x":[],"y":[],"z":[]}
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i],tuple):
                lm_tnp = (verts[measurement_landmarks_inds[i][0]] + 
                          verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm_tnp = verts[measurement_landmarks_inds[i]]
            segments["x"].append(lm_tnp[0])
            segments["y"].append(lm_tnp[1])
            segments["z"].append(lm_tnp[2])
        for ax in ["x","y","z"]:
            segments[ax].append(None)

        if measurement_name in self.measurements:
            m_viz_name = f"{measurement_name}: {self.measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        return go.Scatter3d(x=segments["x"], 
                                    y=segments["y"], 
                                    z=segments["z"],
                                    marker=dict(
                                            size=4,
                                            color="rgba(0,0,0,0)",
                                        ),
                                        line=dict(
                                            color=color,
                                            width=10),
                                        name=m_viz_name
                                        )
        
    def create_measurement_circumference_plot(self,
                                              measurement_name: str,
                                              verts: np.ndarray,
                                              faces: np.ndarray,
                                              color: str):
        '''
        Create circumference measurement plot
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        '''

        circumf_landmarks = self.circumf_definitions[measurement_name]["LANDMARKS"]
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = self.joint2ind[circumf_n1], self.joint2ind[circumf_n2]
        
        plane_origin = np.mean(verts[circumf_landmark_indices,:],axis=0)
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

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
        
        
        draw_segments = {"x":[],"y":[],"z":[]}
        map_ax = {0:"x",1:"y",2:"z"}

        for i in range(slice_segments_hull.shape[0]):
            for j in range(3):
                draw_segments[map_ax[j]].append(slice_segments_hull[i,0,j])
                draw_segments[map_ax[j]].append(slice_segments_hull[i,1,j])
                draw_segments[map_ax[j]].append(None)

        if measurement_name in self.measurements:
            m_viz_name = f"{measurement_name}: {self.measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        return go.Scatter3d(
                            x=draw_segments["x"],
                            y=draw_segments["y"],
                            z=draw_segments["z"],
                            mode="lines",
                            line=dict(
                                color=color,
                                width=10),
                            name=m_viz_name
                                )

    def visualize(self, 
                  measurement_names: List[str] = [], 
                  landmark_names: List[str] = [],
                  title="Measurement visualization"
                  ):
        '''
        Visualize the body model with measurements, landmarks and joints.

        :param measurement_names: List[str], list of strings with measurement names
        :param landmark_names: List[str], list of strings with landmark names
        :param title: str, title of plot
        '''


        fig = go.Figure()

        if self.visualize_body:
            # visualize model mesh
            mesh_plot = self.create_mesh_plot(self.verts, self.faces)
            fig.add_trace(mesh_plot)
            # visualize wireframe
            wireframe_plot = self.create_wireframe_plot(self.verts, self.faces)
            fig.add_trace(wireframe_plot)

        # visualize joints
        if self.visualize_joints:
            joint_plot = self.create_joint_plot(self.joints)
            fig.add_trace(joint_plot)


        # visualize landmarks
        if self.visualize_landmarks:
            landmarks_plot = self.create_landmarks_plot(landmark_names, self.verts)
            fig.add_traces(landmarks_plot)
        

        # visualize measurements
        measurement_colors = dict(zip(self.measurement_types.keys(),
                                  px.colors.qualitative.Alphabet))

        if self.visualize_measurements:
            for m_name in measurement_names:
                if m_name not in self.measurement_types.keys():
                    print(f"Measurement {m_name} not defined.")
                    pass

                if self.measurement_types[m_name] == MeasurementType().LENGTH:
                    measurement_plot = self.create_measurement_length_plot(measurement_name=m_name,
                                                                        verts=self.verts,
                                                                        color=measurement_colors[m_name])     
                elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:
                    measurement_plot = self.create_measurement_circumference_plot(measurement_name=m_name,
                                                                                    verts=self.verts,
                                                                                    faces=self.faces,
                                                                                    color=measurement_colors[m_name])
                
                fig.add_trace(measurement_plot)
                

        fig.update_layout(scene_aspectmode='data',
                            width=1000, height=700,
                            title=title,
                            )
            
        fig.show()


def viz_smplx_joints(visualize_body=True,fig=None,show=True,title="SMPLX joints"):
    """
    Visualize smpl joints on the same plot.
    :param visualize_body: bool, whether to visualize the body or not.
    :param fig: plotly Figure object, if None, create new figure.
    """

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smplx_model =  smplx.create(model_path="data",
                                model_type="smplx",
                                gender="NEUTRAL", 
                                use_face_contour=False,
                                num_betas=10,
                                #body_pose=torch.zeros((1, (55-1) * 3)),
                                ext='pkl')
    
    smplx_model = smplx_model(betas=betas, return_verts=True)
    smplx_joints = smplx_model.joints.detach().numpy()[0]
    smplx_joint_pelvis = smplx_joints[0,:]
    smplx_joints = smplx_joints - smplx_joint_pelvis
    smplx_vertices = smplx_model.vertices.detach().numpy()[0]
    smplx_vertices = smplx_vertices - smplx_joint_pelvis
    smplx_faces = smplx.SMPLX("data/smplx",ext="pkl").faces

    joint_colors = px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   ["#000000"]
    
    if isinstance(fig,type(None)):
        fig = go.Figure()

    for i in range(smplx_joints.shape[0]):

        if i in SMPLX_IND2JOINT.keys():
            joint_name = SMPLX_IND2JOINT[i]
        else:
            joint_name = f"noname-{i}"

        joint_plot = go.Scatter3d(x = [smplx_joints[i,0]],
                                    y = [smplx_joints[i,1]], 
                                    z = [smplx_joints[i,2]], 
                                    mode='markers',
                                    marker=dict(size=10,
                                                color=joint_colors[i],
                                                opacity=1,
                                                symbol="circle"
                                                ),
                                    name="smplx-"+joint_name
                                        )


        fig.add_trace(joint_plot)


    if visualize_body:
        plot_body = go.Mesh3d(
                            x=smplx_vertices[:,0],
                            y=smplx_vertices[:,1],
                            z=smplx_vertices[:,2],
                            color = "red",
                            i=smplx_faces[:,0],
                            j=smplx_faces[:,1],
                            k=smplx_faces[:,2],
                            name='smplx mesh',
                            showscale=True,
                            opacity=0.5
                        )
        fig.add_trace(plot_body)

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title,
                        )
    

    if show:
        fig.show()
    else:
        return fig


def viz_smpl_joints(visualize_body=True,fig=None,show=True,title="SMPL joints"):
    """
    Visualize smpl joints on the same plot.
    :param visualize_body: bool, whether to visualize the body or not.
    :param fig: plotly Figure object, if None, create new figure.
    """

    betas = torch.zeros((1, 10), dtype=torch.float32)
    
    smpl_model =  smplx.create(model_path="data",
                                model_type="smpl",
                                gender="NEUTRAL", 
                                use_face_contour=False,
                                num_betas=10,
                                ext='pkl')
    
    smpl_model = smpl_model(betas=betas, return_verts=True)
    smpl_joints = smpl_model.joints.detach().numpy()[0]
    smpl_joints_pelvis = smpl_joints[0,:]
    smpl_joints = smpl_joints - smpl_joints_pelvis
    smpl_vertices = smpl_model.vertices.detach().numpy()[0]
    smpl_vertices = smpl_vertices - smpl_joints_pelvis
    smpl_faces = smplx.SMPL("data/smpl",ext="pkl").faces


    joint_colors = px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   ["#000000"]
    
    if isinstance(fig,type(None)):
        fig = go.Figure()

    for i in range(smpl_joints.shape[0]):

        if i in SMPL_IND2JOINT.keys():
            joint_name = SMPL_IND2JOINT[i]
        else:
            joint_name = f"noname-{i}"

        joint_plot = go.Scatter3d(x = [smpl_joints[i,0]],
                                    y = [smpl_joints[i,1]], 
                                    z = [smpl_joints[i,2]], 
                                    mode='markers',
                                    marker=dict(size=10,
                                                color=joint_colors[i],
                                                opacity=1,
                                                symbol="cross"
                                                ),
                                    name="smpl-"+joint_name
                                        )


        fig.add_trace(joint_plot)

    if visualize_body:
        plot_body = go.Mesh3d(
                            x=smpl_vertices[:,0],
                            y=smpl_vertices[:,1],
                            z=smpl_vertices[:,2],
                            #facecolor=face_colors,
                            color = "blue",
                            i=smpl_faces[:,0],
                            j=smpl_faces[:,1],
                            k=smpl_faces[:,2],
                            name='smpl mesh',
                            showscale=True,
                            opacity=0.5
                        )
        fig.add_trace(plot_body)

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title,
                        )
    if show:
        fig.show()
    else:
        return fig
                           

def viz_face_segmentation(verts,faces,face_colors,
                          title="Segmented body",name="mesh",show=True):
    """
    Visualize face segmentation defined in face_colors.
    :param verts: np.ndarray - (N,3) representing the vertices
    :param faces: np.ndarray - (F,3) representing the indices of the faces
    :param face_colors: np.ndarray - (F,3) representing the colors of the faces
    """
    
    import plotly.graph_objects as go

    fig = go.Figure()
    mesh_plot = go.Mesh3d(
            x=verts[:,0],
            y=verts[:,1],
            z=verts[:,2],
            facecolor=face_colors,
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            name=name,
            showscale=True,
            opacity=1
        )
    fig.add_trace(mesh_plot)
    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title)
    
    if show:
        fig.show()
    else:
        return fig


def viz_smpl_face_segmentation(fig=None, show=True, title="SMPL face segmentation"):
    body = smplx.SMPL("data/smpl",ext="pkl")

    with open("data/smpl/smpl_body_parts_2_faces.json","r") as f:
        face_segmentation = json.load(f) 

    faces = body.faces
    verts = body.v_template

    # create colors for each face
    colors = px.colors.qualitative.Alphabet + \
            px.colors.qualitative.Dark24
    mapping_bp2ind = dict(zip(face_segmentation.keys(),
                            range(len(face_segmentation.keys()))
                            ))
    face_colors = [0]*faces.shape[0]
    for bp_name,bp_indices in face_segmentation.items():
        bp_label = mapping_bp2ind[bp_name]
        for i in bp_indices:
            face_colors[i] = colors[bp_label]

    if isinstance(fig,type(None)):
        fig = go.Figure()

    fig = viz_face_segmentation(verts,faces,face_colors,title=title,name="smpl",show=False)

    if show:
        fig.show()
    else:
        return fig


def viz_smplx_face_segmentation(fig=None,show=True,title="SMPLX face segmentation"):
    """
    Visualize face segmentations for smplx.
    """
    
    body = smplx.SMPLX("data/smplx",ext="pkl")
    
    with open("data/smplx/smplx_body_parts_2_faces.json","r") as f:
        face_segmentation = json.load(f) 


    faces = body.faces
    verts = body.v_template

    # create colors for each face
    colors = px.colors.qualitative.Alphabet + \
            px.colors.qualitative.Dark24
    mapping_bp2ind = dict(zip(face_segmentation.keys(),
                            range(len(face_segmentation.keys()))
                            ))
    face_colors = [0]*faces.shape[0]
    for bp_name,bp_indices in face_segmentation.items():
        bp_label = mapping_bp2ind[bp_name]
        for i in bp_indices:
            face_colors[i] = colors[bp_label]

    if isinstance(fig,type(None)):
        fig = go.Figure()

    fig = viz_face_segmentation(verts,faces,face_colors,title=title,name="smpl",show=False)

    if show:
        fig.show()
    else:
        return fig


def viz_point_segmentation(verts,point_segm,title="Segmented body",fig=None,show=True):
    """
    Visualze points and their segmentation defined in dict point_segm.
    :param verts: np.ndarray - (N,3) representing the vertices
    :param point_segm: dict - dict mapping body part to all points belonging
                                to it
    """
    colors = px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24
    
    if isinstance(fig,type(None)):
        fig = go.Figure()

    for i, (body_part, body_indices) in enumerate(point_segm.items()):
        plot = go.Scatter3d(x = verts[body_indices,0],
                            y = verts[body_indices,1], 
                            z = verts[body_indices,2], 
                            mode='markers',
                            marker=dict(size=5,
                                        color=colors[i],
                                        opacity=1,
                                        #symbol="cross"
                                        ),
                            name=body_part
                                )
        fig.add_trace(plot)
    fig.update_layout(scene_aspectmode='data',
                    width=1000, height=700,
                    title=title)
    if show:
        fig.show()
    return fig


def viz_smplx_point_segmentation(fig=None,show=True,title="SMPLX point segmentation"):
    """
    Visualize point segmentations for smplx.
    """

    model_path = "data/smplx"
    smpl_verts = smplx.SMPLX(model_path,ext="pkl").v_template
    with open("data/smplx/point_segmentation_meshcapade.json","r") as f:
        point_segm = json.load(f)
    fig = viz_point_segmentation(smpl_verts,point_segm,title=title,fig=fig,show=show)

    if show:
        fig.show()
    else:
        return fig


def viz_smpl_point_segmentation(fig=None,show=True,title="SMPL point segmentation"):
    """
    Visualize point segmentations for smpl.
    """

    model_path = "data/smpl"
    smpl_verts = smplx.SMPL(model_path,ext="pkl").v_template
    with open("data/smpl/point_segmentation_meshcapade.json","r") as f:
        point_segm = json.load(f)
    fig = viz_point_segmentation(smpl_verts,point_segm,title=title,fig=fig,show=show)

    if show:
        fig.show()
    else:
        return fig


def viz_landmarks(verts,landmark_dict,title="Visualize landmarks",fig=None,show=True,name="points"):
    
    if isinstance(fig,type(None)):
        fig = go.Figure()

    plot = go.Scatter3d(x = verts[:,0],
                        y = verts[:,1], 
                        z = verts[:,2], 
                        mode='markers',
                        hovertemplate ='<i>Index</i>: %{text}',
                        text = [i for i in range(verts.shape[0])],
                        marker=dict(size=5,
                                    color="black",
                                    opacity=0.2,
                                    # line=dict(color='black',width=1)
                                    ),
                        name=name
                            )
    
    fig.add_trace(plot)

    colors = px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24  + \
             px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24
    
    for i, (lm_name, lm_ind) in enumerate(landmark_dict.items()):
        plot = go.Scatter3d(x = [verts[lm_ind,0]],
                            y = [verts[lm_ind,1]], 
                            z = [verts[lm_ind,2]], 
                            mode='markers',
                            marker=dict(size=10,
                                        color=colors[i],
                                        opacity=1,
                                        symbol="cross"
                                        ),
                            name=name+"-"+lm_name
                                )
        fig.add_trace(plot)

    fig.update_layout(scene_aspectmode='data',
                    width=1000, height=700,
                    title=title)

    if show:
        fig.show()
    else:
        return fig
    

def viz_smpl_landmarks(fig=None,show=True,title="SMPL landmarks"):
    """
    Visualize smpl landmarks.
    """

    verts = smplx.SMPL("data/smpl",ext="pkl").v_template
    landmark_dict = SMPL_LANDMARK_INDICES

    if isinstance(fig,type(None)):
        fig=go.Figure()

    fig = viz_landmarks(verts,
                        landmark_dict,
                        title="Visualize landmarks",
                        fig=fig,
                        show=show,
                        name="smpl")

    if show:
        fig.show()
    else:
        return fig


def viz_smplx_landmarks(fig=None,show=True,title="SMPLX landmarks"):
    """
    Visualize smplx landmarks.
    """

    verts = smplx.SMPLX("data/smplx",ext="pkl").v_template
    landmark_dict = SMPLX_LANDMARK_INDICES

    if isinstance(fig,type(None)):
        fig=go.Figure()

    fig = viz_landmarks(verts,
                        landmark_dict,
                        title="Visualize landmarks",
                        fig=fig,
                        show=show,
                        name="smplx")

    if show:
        fig.show()
    else:
        return fig

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Visualize body models, joints and segmentations..')
    parser.add_argument('--visualize_smpl_and_smplx_face_segmentation', action='store_true',
                        help="Visualize face segmentations for smplx model.")
    parser.add_argument('--visualize_smpl_and_smplx_joints', action='store_true',
                        help="visualize smpl and smplx joints on same plot.")
    parser.add_argument('--visualize_smpl_and_smplx_point_segmentation', action='store_true',
                        help="visualize smpl and smplx point segmentation on two separate plots.")
    parser.add_argument('--visualize_smpl_and_smplx_landmarks', action='store_true',
                        help="visualize smpl and smplx landmarks on two separate plots.")
    args = parser.parse_args()



    if args.visualize_smpl_and_smplx_face_segmentation:
        # mesh is not compatible with subplots so these are plotted
        # onto separate plots
        viz_smpl_face_segmentation(fig=None, show=True)
        viz_smplx_face_segmentation(fig=None,show=True)


        
    if args.visualize_smpl_and_smplx_joints: 
        title = "SMPL and SMPLX joints"
        fig = viz_smpl_joints(visualize_body=True,
                              fig=None,
                              show=False,
                              title=title)
        viz_smplx_joints(visualize_body=True,
                        fig=fig,
                        show=True,
                        title=title)

    if args.visualize_smpl_and_smplx_point_segmentation:
        fig = make_subplots(rows=1, cols=2, 
                            specs=[[{'type': 'scene'}, 
                                    {'type': 'scene'}]],
                            subplot_titles=("SMPL", "SMPLX")) 
        title="SMPL and SMPLX point segmentation"


        fig_smpl = viz_smpl_point_segmentation(fig=None,show=False,title=title)
        fig_smplx = viz_smplx_point_segmentation(fig=None,show=False,title=title)


        for i in range(len(fig_smpl.data)):
            fig.add_trace(fig_smpl.data[i],row=1,col=1)
        for i in range(len(fig_smplx.data)):
            fig.add_trace(fig_smplx.data[i],row=1,col=2)


        fig.update_layout(fig_smpl.layout)
        fig.update_layout(scene2_aspectmode="data",
                          showlegend=False,
                          width=1200,
                          height=700)
        fig.show()
    
    if args.visualize_smpl_and_smplx_landmarks:
        fig = make_subplots(rows=1, cols=2, 
                            specs=[[{'type': 'scene'}, 
                                    {'type': 'scene'}]],
                            subplot_titles=("SMPL", "SMPLX")) 
        title="SMPL and SMPLX landmarks"

        fig_smpl = viz_smpl_landmarks(fig=None,show=False,title=title)
        fig_smplx = viz_smplx_landmarks(fig=None,show=False,title=title)

        for i in range(len(fig_smpl.data)):
            fig.add_trace(fig_smpl.data[i],row=1,col=1)
        for i in range(len(fig_smplx.data)):
            fig.add_trace(fig_smplx.data[i],row=1,col=2)

        
        fig.update_layout(fig_smpl.layout)
        fig.update_layout(scene2_aspectmode="data",
                          showlegend=False,
                          width=1200,
                          height=700)
        fig.show()