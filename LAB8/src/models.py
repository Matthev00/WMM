import numpy
from moderngl_window.opengl.vao import VAO
import math


def load_quad_2D(program):
    size = 1.0

    positions = []

    positions.append([-size / 2.0, size / 2.0])
    positions.append([-size / 2.0, -size / 2.0])
    positions.append([size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, size / 2.0])
    positions.append([size / 2.0, -size / 2.0])
    positions.append([size / 2.0, size / 2.0])

    positions = numpy.array(positions, dtype=numpy.float32).flatten()

    vao = VAO()
    vao.buffer(positions, "2f", ["in_position"])

    return vao.instance(program)


def load_cube(program):
    size = 1.0

    positions = []
    # Front face
    positions.append([size / 2.0, size / 2.0, size / 2.0])
    positions.append([-size / 2.0, size / 2.0, size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, size / 2.0])
    positions.append([size / 2.0, -size / 2.0, size / 2.0])
    positions.append([size / 2.0, size / 2.0, size / 2.0])
    # Back face
    positions.append([size / 2.0, -size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, size / 2.0, -size / 2.0])
    positions.append([size / 2.0, size / 2.0, -size / 2.0])
    positions.append([size / 2.0, -size / 2.0, -size / 2.0])
    # Top face
    positions.append([size / 2.0, size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, size / 2.0, size / 2.0])
    positions.append([-size / 2.0, size / 2.0, size / 2.0])
    positions.append([size / 2.0, size / 2.0, size / 2.0])
    positions.append([size / 2.0, size / 2.0, -size / 2.0])
    # Bottom face
    positions.append([size / 2.0, -size / 2.0, size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, -size / 2.0])
    positions.append([size / 2.0, -size / 2.0, -size / 2.0])
    positions.append([size / 2.0, -size / 2.0, size / 2.0])
    # Left face
    positions.append([-size / 2.0, size / 2.0, size / 2.0])
    positions.append([-size / 2.0, size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, -size / 2.0])
    positions.append([-size / 2.0, -size / 2.0, size / 2.0])
    positions.append([-size / 2.0, size / 2.0, size / 2.0])
    # Right face
    positions.append([size / 2.0, size / 2.0, -size / 2.0])
    positions.append([size / 2.0, size / 2.0, size / 2.0])
    positions.append([size / 2.0, -size / 2.0, size / 2.0])
    positions.append([size / 2.0, -size / 2.0, size / 2.0])
    positions.append([size / 2.0, -size / 2.0, -size / 2.0])
    positions.append([size / 2.0, size / 2.0, -size / 2.0])

    normals = []
    # Front face
    normals.append([0.0, 0.0, 1.0])
    normals.append([0.0, 0.0, 1.0])
    normals.append([0.0, 0.0, 1.0])
    normals.append([0.0, 0.0, 1.0])
    normals.append([0.0, 0.0, 1.0])
    normals.append([0.0, 0.0, 1.0])
    # Back face
    normals.append([0.0, 0.0, -1.0])
    normals.append([0.0, 0.0, -1.0])
    normals.append([0.0, 0.0, -1.0])
    normals.append([0.0, 0.0, -1.0])
    normals.append([0.0, 0.0, -1.0])
    normals.append([0.0, 0.0, -1.0])
    # Top face
    normals.append([0.0, 1.0, 0.0])
    normals.append([0.0, 1.0, 0.0])
    normals.append([0.0, 1.0, 0.0])
    normals.append([0.0, 1.0, 0.0])
    normals.append([0.0, 1.0, 0.0])
    normals.append([0.0, 1.0, 0.0])
    # Bottom face
    normals.append([0.0, -1.0, 0.0])
    normals.append([0.0, -1.0, 0.0])
    normals.append([0.0, -1.0, 0.0])
    normals.append([0.0, -1.0, 0.0])
    normals.append([0.0, -1.0, 0.0])
    normals.append([0.0, -1.0, 0.0])
    # Left face
    normals.append([-1.0, 0.0, 0.0])
    normals.append([-1.0, 0.0, 0.0])
    normals.append([-1.0, 0.0, 0.0])
    normals.append([-1.0, 0.0, 0.0])
    normals.append([-1.0, 0.0, 0.0])
    normals.append([-1.0, 0.0, 0.0])
    # Right face
    normals.append([1.0, 0.0, 0.0])
    normals.append([1.0, 0.0, 0.0])
    normals.append([1.0, 0.0, 0.0])
    normals.append([1.0, 0.0, 0.0])
    normals.append([1.0, 0.0, 0.0])
    normals.append([1.0, 0.0, 0.0])

    # Alternative to normals generation
    # generate_normals(positions)

    positions = numpy.array(positions, dtype=numpy.float32).flatten()
    normals = numpy.array(normals, dtype=numpy.float32).flatten()

    vao = VAO()
    vao.buffer(positions, "3f", ["in_position"])
    vao.buffer(normals, "3f", ["in_normal"])
    return vao.instance(program)


def load_cone(program, radius=0.75, height=1.0, segments=32, size=1.0):
    positions = []
    normals = []
    half_height = height * size / 2.0
    radius = radius * size

    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        next_theta = 2.0 * math.pi * (i + 1) / segments

        x1 = radius * math.cos(theta)
        z1 = radius * math.sin(theta)
        x2 = radius * math.cos(next_theta)
        z2 = radius * math.sin(next_theta)

        # Base face (Triangle fan around the base center)
        positions.extend(
            [[x1, -half_height, z1], [x2, -half_height, z2], [0.0, -half_height, 0.0]]
        )
        normals.extend([[0.0, -1.0, 0.0]] * 3)

        # Side face (Triangles connecting the base to the apex)
        apex_normal = [0.0, 1.0, 0.0]  # Normal at the apex pointing up
        positions.extend(
            [[0.0, half_height, 0.0], [x2, -half_height, z2], [x1, -half_height, z1]]
        )

        # Calculate normals for the side faces
        nx1, nz1 = x1, z1
        nx2, nz2 = x2, z2
        normal = [nx1 + nx2, height, nz1 + nz2]
        normal_length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        normal = [n / normal_length for n in normal]
        normals.extend([normal, normal, normal])

    positions = numpy.array(positions, dtype=numpy.float32).flatten()
    normals = numpy.array(normals, dtype=numpy.float32).flatten()

    vao = VAO()
    vao.buffer(positions, "3f", ["in_position"])
    vao.buffer(normals, "3f", ["in_normal"])
    return vao.instance(program)


def load_cylinder(program, radius=0.75, height=1.0, segments=32, size=1.2):
    positions = []
    normals = []
    half_height = height * size / 2.0
    radius = radius * size

    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        next_theta = 2.0 * math.pi * (i + 1) / segments

        x1 = radius * math.cos(theta)
        z1 = radius * math.sin(theta)
        x2 = radius * math.cos(next_theta)
        z2 = radius * math.sin(next_theta)

        # Top face
        positions.extend(
            [[x1, half_height, z1], [0.0, half_height, 0.0], [x2, half_height, z2]]
        )
        normals.extend([[0.0, 1.0, 0.0]] * 3)

        # Bottom face
        positions.extend(
            [[x1, -half_height, z1], [x2, -half_height, z2], [0.0, -half_height, 0.0]]
        )
        normals.extend([[0.0, -1.0, 0.0]] * 3)

        # Front side face
        positions.extend(
            [[x1, half_height, z1], [x1, -half_height, z1], [x2, half_height, z2]]
        )
        normals.extend([[x1, 0.0, z1]] * 3)

        positions.extend(
            [[x1, -half_height, z1], [x2, -half_height, z2], [x2, half_height, z2]]
        )
        normals.extend([[x2, 0.0, z2]] * 3)

        # Back side face
        positions.extend(
            [[x2, half_height, z2], [x1, -half_height, z1], [x1, half_height, z1]]
        )
        normals.extend([[x1, 0.0, -z1]] * 3)

        positions.extend(
            [[x2, -half_height, z2], [x1, -half_height, z1], [x2, half_height, z2]]
        )
        normals.extend([[x2, 0.0, -z2]] * 3)

    positions = numpy.array(positions, dtype=numpy.float32).flatten()
    normals = numpy.array(normals, dtype=numpy.float32).flatten()

    vao = VAO()
    vao.buffer(positions, "3f", ["in_position"])
    vao.buffer(normals, "3f", ["in_normal"])
    return vao.instance(program)


def generate_normals(positions):
    N = len(positions)
    normals = []
    for i in range(0, N, 3):
        p0 = numpy.array(positions[i + 0])
        p1 = numpy.array(positions[i + 1])
        p2 = numpy.array(positions[i + 2])
        cross = numpy.cross(p1 - p0, p2 - p0)
        norm = cross / numpy.linalg.norm(cross)
        for v in range(3):
            normals.append(list(norm))
    return normals
