# Cosmic Vine v1 - Luis De Cunto
#
# The file creates a "vine" modeled as a cube sequence.
# Each new cube originates from the face of the previous one, 
# with a random rotation
#
# This version DOES NOT check for self intersection.
# One way of implementing it would be to check the minimum distance 
# between skeleton segments (1-D lines between centers of the cubes)
# and set up a threshold for that distance. It would treat the cubes as
# spheres but is a possible approximation.
#
# Self-intersection detection (sphere approximation):
#
# for each candidate cube:
#     new_segment = (current_center, candidate_center)
#     for each previous_segment in skeleton:
#         dist = min_distance_between_segments(new_segment, previous_segment)
#         if dist < threshold:
#             reject candidate  # too close, would overlap
#     if not rejected:
#         accept candidate
#
# Threshold ≈ side_length treats each cube as its inscribed sphere
# (radius = side_length/2). Two spheres overlap when their centers
# are closer than the sum of their radii (side_length).
# This allows minor corner/edge overlap but prevents bulk intersection.

import numpy as np
import random

### INPUT PARAMETERS
side_length = 1.0                           # Side length of the cubes
C0 = np.array([0.0, 0.0, 0.0])              # Center of the first cube
ROT0 = np.array([0.0, 0.0, 0.0])            # Rotation angles of the first cube (with respect of global axes)
n_gen = 100                                 # How many cubes or "generations" does the sequence run
fname = f"cosmic_vine_{n_gen}_cubes.obj"    # Name of the output object


### Helper functions
def unit_cube(side_length):
    # Unit cube vertices
    
    s = side_length / 2.0
    return np.array([
        [-s, -s, -s],  # 0
        [+s, -s, -s],  # 1
        [+s, +s, -s],  # 2
        [-s, +s, -s],  # 3
        [-s, -s, +s],  # 4
        [+s, -s, +s],  # 5
        [+s, +s, +s],  # 6
        [-s, +s, +s],  # 7
    ])

def rotation_about_axis(axis, angle_deg):
    # Rodrigues' formula: 3x3 rotation matrix for a given axis and angle (degrees).
    # Ref: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    # Used for getting the local csys of the next cube, given the local csys of the 
    # previous one, by rotating around the axis that they share.

    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array([[    0, -axis[2],  axis[1]],
                  [ axis[2],     0, -axis[0]],
                  [-axis[1],  axis[0],     0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def generate_new_cube(cube_prev):
    # Generate new cube based on the previous (parent) cube.

    # Choose random face (exclude parent_face)
    available = [f for f in range(6)
                 if f != cube_prev.parent_face and f != cube_prev.child_face]
    chosen_face = random.choice(available)
    cube_prev.child_face = chosen_face

    # Find the new cube's center: face center + side_length/2 along outward normal
    face_center, normal = cube_prev.get_face_center_and_normal(chosen_face)
    new_center = face_center + normal * (cube_prev.side_length / 2.0)

    # Rotate around the face normal by a random angle
    spin_angle = random.uniform(-180, 180)
    R_spin = rotation_about_axis(normal, spin_angle)

    # New cube's rotation = spin around normal composed with parent's rotation
    new_rotation = R_spin @ cube_prev.rotation

    # The child's parent_face is the opposite of the chosen face
    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    child_parent_face = OPPOSITE[chosen_face]

    # New cube!
    cube_new = Cube(cube_id = cube_prev.cube_id+1,
                    side_length=cube_prev.side_length,
                    center = new_center,
                    rotation_matrix = new_rotation,
                    parent_face = child_parent_face)

    return cube_new

def plot_cube(ax,cube):
    # Auxiliar function to plot cubes while debugging (not needed for run the code)
    for p0, p1 in cube.edges():
        ax.plot3D(*zip(p0, p1), color='black')

def export_obj(cubes, filename):
    # Export the cube as an obj, defining faces and vertices

    FACE_VERTEX_INDICES = [
        [1, 2, 6, 5],  # 0: +X
        [0, 4, 7, 3],  # 1: -X
        [2, 3, 7, 6],  # 2: +Y
        [0, 1, 5, 4],  # 3: -Y
        [4, 5, 6, 7],  # 4: +Z
        [0, 3, 2, 1],  # 5: -Z
    ]

    vertex_offset = 0
    with open(filename, 'w') as f:
        f.write("# Cosmic Vine\n")
        for cube in cubes:
            f.write(f"# Cube {cube.cube_id}\n")
            for v in cube.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face_verts in FACE_VERTEX_INDICES:
                indices = [vi + 1 + vertex_offset for vi in face_verts]
                f.write(f"f {indices[0]} {indices[1]} {indices[2]} {indices[3]}\n")
            vertex_offset += 8

# Cube Class 
class Cube:
    # Cube class, contains:
    # - cube_id
    # - side_length
    # - local_csys: contains local csys of the cube (center and orientation) 
    # - rotation: Global rotation matrix
    # - faces
    # - vertices
    # - parent_cube: previous cube
    # - parent_face: face in contact with the previous cube
    # - child_face; face in contact with the next cube

    def __init__(self, cube_id, side_length, center=np.array([0.0, 0.0, 0.0]),
                 rotation_angles=np.array([0.0, 0.0, 0.0]), rotation_matrix=None,
                 parent_cube=None, parent_face=None):
        
        self.cube_id = cube_id
        self.side_length = side_length
        self.parent_cube = parent_cube
        self.parent_face = parent_face
        self.child_face = None
        self.vertices = []
        self.faces = [0, 1, 2, 3, 4, 5]

        if rotation_matrix is not None:
            self.rotation = rotation_matrix
        else:
            # Build rotation matrix from Euler angles (in degrees)
            ax, ay, az = np.radians(rotation_angles)
            Rx = np.array([[1, 0, 0],
                            [0, np.cos(ax), -np.sin(ax)],
                            [0, np.sin(ax),  np.cos(ax)]])
            Ry = np.array([[ np.cos(ay), 0, np.sin(ay)],
                            [ 0,          1, 0         ],
                            [-np.sin(ay), 0, np.cos(ay)]])
            Rz = np.array([[np.cos(az), -np.sin(az), 0],
                            [np.sin(az),  np.cos(az), 0],
                            [0,           0,          1]])
            self.rotation = Rz @ Ry @ Rx

        # Build 4x4 local-to-global transform
        self.local_csys = np.eye(4)
        self.local_csys[:3, :3] = self.rotation
        self.local_csys[:3, 3] = center

        self.compute_vertices()

    def compute_vertices(self):
        local_verts = unit_cube(self.side_length)
        local_hom = np.hstack([local_verts, np.ones((8, 1))])
        global_hom = (self.local_csys @ local_hom.T).T
        self.vertices = global_hom[:, :3]

    def edges(self):
        edge_pairs = [
            (0,1),(1,2),(2,3),(3,0),  # back face
            (4,5),(5,6),(6,7),(7,4),  # front face
            (0,4),(1,5),(2,6),(3,7),  # connecting edges
        ]
        return [(self.vertices[i], self.vertices[j]) for i, j in edge_pairs]

    def face_vertices(self, face_index):
        face_map = {
            0: [1, 2, 6, 5],  # +X
            1: [0, 4, 7, 3],  # -X
            2: [2, 3, 7, 6],  # +Y
            3: [0, 1, 5, 4],  # -Y
            4: [4, 5, 6, 7],  # +Z
            5: [0, 3, 2, 1],  # -Z
        }
        return [self.vertices[i] for i in face_map[face_index]]

    def get_face_center_and_normal(self, face_index):
        FACE_NORMALS = {
            0: np.array([+1,  0,  0]),  # +X
            1: np.array([-1,  0,  0]),  # -X
            2: np.array([ 0, +1,  0]),  # +Y
            3: np.array([ 0, -1,  0]),  # -Y
            4: np.array([ 0,  0, +1]),  # +Z
            5: np.array([ 0,  0, -1]),  # -Z
        }
        normal_local = FACE_NORMALS[face_index]
        center_local = normal_local * (self.side_length / 2.0)

        # Transform center as a point (with translation)
        center_hom = np.array([*center_local, 1.0])
        center_global = (self.local_csys @ center_hom)[:3]

        # Transform normal as a direction (rotation only, no translation)
        normal_global = self.local_csys[:3, :3] @ normal_local

        return center_global, normal_global

if __name__ == "__main__":
    cube0 = Cube(0,side_length=side_length,center=C0, rotation_angles=ROT0)
    vine = [cube0]
    for i_cube in range(n_gen-1):
        cube_new = generate_new_cube(vine[i_cube])
        vine.append(cube_new)
        c0 = vine[i_cube].local_csys[:3, 3]
        c1 = cube_new.local_csys[:3, 3]

export_obj(vine, fname)
