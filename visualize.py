
import numpy as np
from typing import List
import plotly
import plotly.graph_objects as go
import plotly.express as px
import trimesh

from measurement_definitions import MeasurementType
from utils import convex_hull_from_3D_points, filter_body_part_slices

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