import argparse, sys, os, math, re
import bpy
from mathutils import Vector, Matrix
import mathutils
import numpy as np
import json 
import random 
import glob
import objaverse 

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--views', type=int, default=24,
    help='number of views to be rendered')
parser.add_argument(
    '--obj', type=str,
    help='Path to the obj file to be rendered.')
parser.add_argument(
    '--obj_id', type=str,
    help='Objaverse object ID to render (alternative to --obj)')
parser.add_argument(
    '--output', type=str, default=None,
    help='The path the output will be dumped to.')
parser.add_argument(
    '--format', type=str, default='PNG',
    help='Format of files generated. Either PNG or OPEN_EXR')

parser.add_argument(
    '--input_model', type=str, default='glb',
    help='glb is the format for objaverse, but we can use [obj,ply,glb]')
parser.add_argument(
    '--outf_name', type=str, default=None,
    help='folder to put things in')

parser.add_argument(
    '--depth', action="store_true",
    help='save_depth')

parser.add_argument(
    '--normal', action="store_true",
    help='save_normal')


parser.add_argument(
    '--use_model_identifier', action='store_true',
    help='add the name of the folder to the end of the thing.')

parser.add_argument(
    '--add_floor', action='store_true',
    help='add_plain_floor')

parser.add_argument(
    '--assets_hdri',
    default = None,
    help = "Path to a set of hdr to select randomly for illumination."
    )


parser.add_argument(
    '--resolution', type=int, default=256,
    help='Resolution of the images.')
parser.add_argument(
    '--engine', type=str, default='CYCLES',
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Handle obj_id if provided
if args.obj_id is not None:
    if args.obj is not None:
        raise ValueError("Cannot specify both --obj and --obj_id")
    print(f"Loading object from Objaverse with ID: {args.obj_id}")
    objects = objaverse.load_objects(uids=[args.obj_id])
    if not objects or args.obj_id not in objects:
        raise ValueError(f"Could not find object with ID: {args.obj_id}")
    args.obj = objects[args.obj_id]
    print(f"Found object path: {args.obj}")

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

# bpy.ops.object.delete()
# bpy.ops.objects['Light'].delete()
# bpy.data.objects['Light'].select_set(True)
# bpy.ops.object.delete()

render.engine = args.engine
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
bpy.context.scene.cycles.filter_width = 0.01
# bpy.context.scene.render.film_transparent = True

bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.diffuse_bounces = 1
bpy.context.scene.cycles.glossy_bounces = 1
bpy.context.scene.cycles.transparent_max_bounces = 3
bpy.context.scene.cycles.transmission_bounces = 3
bpy.context.scene.cycles.samples = 64
bpy.context.scene.cycles.use_denoising = True

scene.world.use_nodes = True
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'


def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Compute device selected: {0}".format(compute_device_type))
            break
        except TypeError:
            pass

    # Any CUDA/OPENCL devices?
    acceleratedTypes = ['CUDA', 'OPENCL']
    accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
    print('Accelerated render = {0}'.format(accelerated))

    # If we have CUDA/OPENCL devices, enable only them, otherwise enable
    # all devices (assumed to be CPU)
    print(cprefs.devices)
    for device in cprefs.devices:
        device.use = not accelerated or device.type in acceleratedTypes
        print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))

    return accelerated


enable_cuda_devices()
context.active_object.select_set(True)
bpy.ops.object.delete()

bpy.data.objects['Light'].select_set(True)
bpy.ops.object.delete()


# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')


def bounds(obj, local=False):
    local_coords = obj.bound_box[:]
    om = obj.matrix_world

    if not local:
        worldify = lambda p: om @ Vector(p[:])
        coords = [worldify(p).to_tuple() for p in local_coords]
    else:
        coords = [p[:] for p in local_coords]

    rotated = zip(*coords[::-1])

    push_axis = []
    for (axis, _list) in zip('xyz', rotated):
        info = lambda: None
        info.max = max(_list)
        info.min = min(_list)
        info.distance = info.max - info.min
        push_axis.append(info)

    import collections

    originals = dict(zip(['x', 'y', 'z'], push_axis))

    o_details = collections.namedtuple('object_details', 'x y z')
    return o_details(**originals)

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

# imported_object = bpy.ops.import_scene.obj(filepath=args.obj, use_edges=False, use_smooth_groups=False, split_mode='OFF')
if args.input_model == "glb":
    imported_object = bpy.ops.import_scene.gltf(filepath=args.obj)
elif args.input_model == "ply":
    imported_object = bpy.ops.import_mesh.ply(filepath=args.obj)
    obj_object = bpy.context.selected_objects[0]
    print('Imported name:', obj_object.name)
    bpy.data.objects[obj_object.name].select_set(True)
    bpy.ops.paint.vertex_paint_toggle()

    #bpy.context.area.ui_type = 'ShaderNodeTree'

    #bpy.ops.material.new()

    mat = bpy.data.materials.get("Material")

    if len(bpy.context.active_object.data.materials) == 0:
        bpy.context.active_object.data.materials.append(bpy.data.materials['Material'])
    else:
        bpy.context.active_object.data.materials[0] = bpy.data.materials['Material']
    if mat:
        mat.node_tree.nodes.new("ShaderNodeVertexColor")
        mat.node_tree.links.new(mat.node_tree.nodes[2].outputs['Color'], mat.node_tree.nodes[1].inputs['Base Color'])


else:
    bpy.ops.import_scene.obj(filepath=args.obj, use_edges=False, use_smooth_groups=False, split_mode='OFF',axis_up='Z')

# for this_obj in bpy.data.objects:
#     if this_obj.type == "MESH":
#         this_obj.select_set(True)
#         bpy.context.view_layer.objects.active = this_obj
#         bpy.ops.object.mode_set(mode='EDIT')
#         bpy.ops.mesh.split_normals()

bpy.ops.object.mode_set(mode='OBJECT')
print(len(bpy.context.selected_objects))
obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

# taken from https://devtalk.blender.org/t/calculating-center-point-of-all-mesh-in-scene/18095
def calcBoundingBox(mesh_objs):
    cornerApointsX = []
    cornerApointsY = []
    cornerApointsZ = []
    cornerBpointsX = []
    cornerBpointsY = []
    cornerBpointsZ = []
    
    for ob in mesh_objs:
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bbox_corners = [ob.matrix_world @ Vector(corner)  for corner in ob.bound_box]
        cornerApointsX.append(bbox_corners[0].x)
        cornerApointsY.append(bbox_corners[0].y)
        cornerApointsZ.append(bbox_corners[0].z)
        cornerBpointsX.append(bbox_corners[6].x)
        cornerBpointsY.append(bbox_corners[6].y)
        cornerBpointsZ.append(bbox_corners[6].z)
        
    minA = Vector((min(cornerApointsX), min(cornerApointsY), min(cornerApointsZ)))
    maxB = Vector((max(cornerBpointsX), max(cornerBpointsY), max(cornerBpointsZ)))

    center_point = Vector(((minA.x + maxB.x)/2, (minA.y + maxB.y)/2, (minA.z + maxB.z)/2))
    dimensions =  Vector((maxB.x - minA.x, maxB.y - minA.y, maxB.z - minA.z))
    
    return center_point, dimensions

mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH'] 
center_point, dimensions = calcBoundingBox(mesh_objs)
print(center_point,dimensions)


for obj in bpy.data.objects:
    if obj.type == 'MESH':
        while obj.parent is not None:
            obj = obj.parent
            print(obj)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.location_clear(clear_delta=False)
        scale = 1/max(dimensions)*0.5
        setattr(obj, 'scale', [scale,scale,scale])
        setattr(obj, 'location', obj.location-(center_point*scale))

        # just need to find that one parent object (coming from the mesh)
        break

# print(dimensions)
# raise()
if args.add_floor:
    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(0, 0, -scale * dimensions[2]/2 + 0.001), 
        scale=(1, 1, 1)
    )
    bpy.context.view_layer.update()
# scaling_value = random.uniform(1, 3)
# bpy.context.object.scale = (scaling_value,scaling_value,1)

# HDR LIGHT
if not args.assets_hdri is None:
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']

    node_environment = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property
    skyboxes = glob.glob(f'{args.assets_hdri}/*.hdr')
    skybox_random_selection = skyboxes[random.randint(0,len(skyboxes)-1)]

    node_environment.image = bpy.data.images.load(skybox_random_selection) # Relative path
    # node_environment.location = -300,0

    world.node_tree.links.new(node_environment.outputs["Color"], bg.inputs["Color"])

else:

# bg.inputs[0].default_value[:3] = (1, 1, 1)
# bg.inputs[1].default_value = 1.0

    # # add a light above the object 
    bpy.ops.object.light_add(type='AREA')
    light2 = bpy.data.lights['Area']

    light2.energy = 30000
    bpy.data.objects['Area'].location[1] = -1
    bpy.data.objects['Area'].scale[0] = 100
    bpy.data.objects['Area'].scale[1] = 100
    bpy.data.objects['Area'].scale[2] = 100


world = bpy.data.worlds['World']
world.use_nodes = True
bg_node = world.node_tree.nodes['Background']
bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
bg_node.inputs[1].default_value = 1.0  # Full strength


# Place camera

cam = scene.objects['Camera']
cam.location = (0, 1.2, 0)  # radius equals to 1
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty



# Function taken from https://github.com/zhenpeiyang/HM3D-ABO/blob/master/my_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = mathutils.Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

# Function taken from https://github.com/zhenpeiyang/HM3D-ABO/blob/master/my_blender.py
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = mathutils.Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# function taken from https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at(obj, target, roll=0):
    """
    Rotate obj to look at target

    :arg obj: the object to be rotated. Usually the camera
    :arg target: the location (3-tuple or Vector) to be looked at
    :arg roll: The angle of rotation about the axis from obj to target in radians. 

    Based on: https://blender.stackexchange.com/a/5220/12947 (ideasman42)      
    """
    if not isinstance(target, mathutils.Vector):
        target = mathutils.Vector(target)
    loc = obj.location
    # direction points from the object to the target
    direction = target - loc
    tracker, rotator = (('-Z', 'Y'),'Z') if obj.type=='CAMERA' else (('X', 'Z'),'Y') #because new cameras points down(-Z), usually meshes point (-Y)
    quat = direction.to_track_quat(*tracker)
    
    # /usr/share/blender/scripts/addons/add_advanced_objects_menu/arrange_on_curve.py
    quat = quat.to_matrix().to_4x4()
    rollMatrix = mathutils.Matrix.Rotation(roll, 4, rotator)

    # remember the current location, since assigning to obj.matrix_world changes it
    loc = loc.to_tuple()
    #obj.matrix_world = quat * rollMatrix
    # in blender 2.8 and above @ is used to multiply matrices
    # using * still works but results in unexpected behaviour!
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc


def sphere_renders(
        nb_planes,
        nb_circle,
        elevation_range = [0,180],
        tetha_range = [0,360]
    ):

    positions_to_render = []
    for i_plane in range(nb_planes):
        elevation = np.deg2rad(  elevation_range[0] + \
                                ((i_plane+1) * (elevation_range[1]-elevation_range[0])/(nb_planes+1)))
        for i_circle in range(nb_circle):
            azimuth = np.deg2rad(tetha_range[0]+((i_circle+1) * (tetha_range[1]-tetha_range[0])/(nb_circle+1)))
            eye_position = [
                np.sin(elevation)*np.cos(azimuth),
                np.sin(elevation)*np.sin(azimuth),
                np.cos(elevation),
            ]
            positions_to_render.append(eye_position)
    return positions_to_render


def random_sample_sphere(
        elevation_range = [0,180],
        tetha_range = [0,360],
        nb_frames = 10,
    ):
    to_return = []
    outside = True
    max_radius = 1.00001
    min_radius = 0.99999
    min_max_x = [0,0]
    min_max_y = [0,0]
    min_max_z = [0,0]
    for i_degree in range(tetha_range[0],tetha_range[1],1):
        v = np.cos(np.deg2rad(i_degree))
        if v < min_max_x[0]:
            min_max_x[0] = v
        if v > min_max_x[1]:
            min_max_x[1] = v

    for i_degree in range(tetha_range[0],tetha_range[1],1):
        v = np.sin(np.deg2rad(i_degree))
        if v < min_max_y[0]:
            min_max_y[0] = v
        if v > min_max_y[1]:
            min_max_y[1] = v

    for i_degree in range(elevation_range[0],elevation_range[1],1):
        v = np.cos(np.deg2rad(i_degree))
        if v < min_max_z[0]:
            min_max_z[0] = v
        if v > min_max_z[1]:
            min_max_z[1] = v

    for i in range(nb_frames):
        outside = True
        while outside:

            x = random.uniform(min_max_x[0], min_max_x[1])
            y = random.uniform(min_max_y[0], min_max_y[1])
            z = random.uniform(min_max_z[0], min_max_z[1])

            if  (x**2 + y**2 + z**2) * max_radius < max_radius + 0.0001 \
            and (x**2 + y**2 + z**2) * max_radius > min_radius:
                outside = False
        to_return.append([x,y,z])
    return to_return

positions = sphere_renders(
      nb_planes = 1, 
      nb_circle = args.views,
      elevation_range = [55,65],
      tetha_range = [0,359]
  )

# positions = random_sample_sphere(
#         elevation_range = [2,188],
#         tetha_range = [0,360],
#         nb_frames = args.views,
#     )

# bpy.context.scene.render.filepath = f'{path}/{000}.png'

# bpy.ops.render.render(write_still = True)

obj_camera = bpy.data.objects["Camera"]

look_at_pos = mathutils.Vector((0,0,0))
# print(bpy.data.cameras[0].angle_x)
bpy.data.cameras[0].angle_x = 0.6911112070083618


bpy.context.scene.render.resolution_x = args.resolution
bpy.context.scene.render.resolution_y = args.resolution


K = get_calibration_matrix_K_from_blender(bpy.data.cameras[0])

if args.use_model_identifier:
    model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
    path = os.path.join(os.path.abspath(args.output), model_identifier)
else:
    path = os.path.abspath(args.output)

if not args.outf_name is None:
    path = os.path.join(os.path.abspath(args.output), args.outf_name)

to_export = {
    # 'fx': K[0][0],
    # 'fy': K[1][1],
    # 'cx': K[0][-1],
    # 'cy': K[1][-1],
    'camera_angle_x': bpy.data.cameras[0].angle_x,
    "aabb": [
        [
            0.5,
            0.5,
            0.5
        ],
        [
            -0.5,
            -0.5,
            -0.5
        ]
    ],
}

frames = []


# add the depth if needed
if args.depth: 
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)
    
    bpy.context.scene.render.image_settings.use_zbuffer=True
    bpy.context.view_layer.use_pass_z=True

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    depth_file_output.format.file_format = "OPEN_EXR"
    depth_file_output.base_path = ''

    node_viewer = tree.nodes.new('CompositorNodeViewer') 
    node_viewer.use_alpha = False  
    links.new(render_layers.outputs['Image'], node_viewer.inputs[0])    

if args.normal: 
    if not args.depth: 
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        for n in tree.nodes:
            tree.nodes.remove(n)
        # Create input render layer node.
        render_layers = tree.nodes.new('CompositorNodeRLayers')
    
    bpy.context.view_layer.use_pass_normal=True
    bpy.context.view_layer.cycles.denoising_store_passes=True

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'



    links.new(render_layers.outputs['Denoising Normal'], normal_file_output.inputs[0])
    normal_file_output.format.file_format = "OPEN_EXR"
    normal_file_output.base_path = ''

# bpy.ops.wm.save_as_mainfile(filepath=f'{path}/scene.blend')

# Check if PNG files already exist and skip rendering if they do
existing_png_files = glob.glob(f'{path}/*.png')
if existing_png_files:
    print(f"PNG files already exist in {path}, skipping rendering")
    # Still need to create transforms.json with the expected frames
    for i_pos in range(len(positions)):
        to_add = {
            "file_path": f'{str(i_pos).zfill(3)}.png',
            "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # Identity matrix as placeholder
        }
        frames.append(to_add)
    to_export['frames'] = frames
    with open(f'{path}/transforms.json', 'w') as f:
        json.dump(to_export, f, indent=4)
    sys.exit(0)

for i_pos, pos in enumerate(positions):

    obj_camera.location = (
        (pos[0])+look_at_pos[0],
        (pos[1])+look_at_pos[1],
        (pos[2])+look_at_pos[2]
        )
    # print(obj_camera.location)
    # look_at(obj_camera,mathutils.Vector((0,0,0)))
    look_at(obj_camera,look_at_pos)

    bpy.context.view_layer.update()

    rt = get_3x4_RT_matrix_from_blender(obj_camera)
    pos, rt, scale = obj_camera.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for i in range(3):
        a = []
        for j in range(3):
            a.append(rt[i][j])
        a.append(pos[i])
        matrix.append(a)
    matrix.append([0,0,0,1])
    # print(matrix)

    # print(obj_camera.matrix_world.decompose())
    # raise()
    # matrix[0][-1]=obj_camera.location[0]
    # matrix[1][-1]=obj_camera.location[1]
    # matrix[2][-1]=obj_camera.location[2]
    # print(matrix)

    to_add = {\
        "file_path":f'{str(i_pos).zfill(3)}.png',
        "transform_matrix":matrix
    }
    frames.append(to_add)

    if args.depth: 
        depth_file_output.file_slots[0].path = f'{path}/{str(i_pos).zfill(3)}_depth'
    if args.normal:
        normal_file_output.file_slots[0].path = f'{path}/{str(i_pos).zfill(3)}_normal'

    bpy.context.scene.render.filepath = f'{path}/{str(i_pos).zfill(3)}.png'
    bpy.ops.render.render(write_still = True)
    print(f"Saved the image to {path}/{str(i_pos).zfill(3)}.png")

    if args.depth:
        os.rename(f'{path}/{str(i_pos).zfill(3)}_depth{str(bpy.data.scenes[0].frame_current).zfill(4)}.exr', f'{path}/{str(i_pos).zfill(3)}_depth.exr') 
    if args.normal:
        os.rename(f'{path}/{str(i_pos).zfill(3)}_normal{str(bpy.data.scenes[0].frame_current).zfill(4)}.exr', f'{path}/{str(i_pos).zfill(3)}_normal.exr') 
    # raise()
    # time.sleep(10)
    # break

    to_export['frames'] = frames

with open(f'{path}/transforms.json', 'w') as f:
    json.dump(to_export, f,indent=4)
