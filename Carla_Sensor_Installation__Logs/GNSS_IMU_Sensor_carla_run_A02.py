import carla
import random
import time
import pygame
import numpy as np
import matplotlib.cm as cm
import csv 
import os 
from datetime import datetime


# ============================================================
# CARLA CONNECTION
# ============================================================

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)

world = client.get_world()

original_settings = world.get_settings()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)


# ============================================================
# DESTROY EXISTING VEHICLES
# ============================================================

for actor in world.get_actors().filter("vehicle.*"):
    try:
        actor.destroy()
    except RuntimeError:
        pass


blueprint_library = world.get_blueprint_library()


# ============================================================
# TRAFFIC MANAGER
# ============================================================

traffic_manager = client.get_trafficmanager(8000)

traffic_manager.set_synchronous_mode(True)

traffic_manager.set_global_distance_to_leading_vehicle(
    8.0
)

traffic_manager.global_percentage_speed_difference(
    30.0
)

print("Synchronous simulation enabled")


# ============================================================
# SPAWN POINTS
# ============================================================

spawn_points = world.get_map().get_spawn_points()

print(
    "Available spawn points:",
    len(spawn_points)
)


# ============================================================
# EGO VEHICLE
# ============================================================

vehicle_bp = blueprint_library.find(
    "vehicle.bmw.grandtourer"
)

my_spawn_point = spawn_points[12]

my_vehicle = world.try_spawn_actor(
    vehicle_bp,
    my_spawn_point
)

if my_vehicle is None:
    raise RuntimeError(
        "Failed to spawn ego vehicle"
    )

print(
    "Ego vehicle:",
    my_vehicle.type_id
)

# Ego vehicle is MANUAL
my_vehicle.set_autopilot(False)


# ============================================================
# TRAFFIC
# ============================================================

traffic_vehicles = []

traffic_blueprints = blueprint_library.filter(
    "vehicle.*"
)


# ------------------------------------------------------------
# Use shuffled spawn points
# ------------------------------------------------------------

available_spawn_points = spawn_points.copy()

random.shuffle(
    available_spawn_points
)


# ------------------------------------------------------------
# Spawn traffic
# ------------------------------------------------------------

for spawn_point in available_spawn_points:

    # Don't spawn too close to ego
    if spawn_point.location.distance(
        my_spawn_point.location
    ) < 30.0:

        continue


    # Select a vehicle
    vehicle_bp = random.choice(
        traffic_blueprints
    )


    # Spawn
    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
    )


    if vehicle is None:
        continue


    # ========================================================
    # TRAFFIC MANAGER CONTROL
    # ========================================================

    vehicle.set_autopilot(
        True,
        traffic_manager.get_port()
    )

    # Keep large distance
    traffic_manager.distance_to_leading_vehicle(
        vehicle,
        8.0
    )

    # Drive slower
    traffic_manager.vehicle_percentage_speed_difference(
        vehicle,
        30.0
    )

    # Don't randomly change lanes
    traffic_manager.auto_lane_change(
        vehicle,
        False
    )

    # Disable random lane changes
    traffic_manager.random_left_lanechange_percentage(
        vehicle,
        0.0
    )

    traffic_manager.random_right_lanechange_percentage(
        vehicle,
        0.0
    )
    # ========================================================
    # TRAFFIC BEHAVIOR
    # ========================================================




    # IMPORTANT:
    # Don't randomly change lanes
    traffic_manager.auto_lane_change(
        vehicle,
        False
    )


    # No random left lane changes
    traffic_manager.random_left_lanechange_percentage(
        vehicle,
        0.0
    )


    # No random right lane changes
    traffic_manager.random_right_lanechange_percentage(
        vehicle,
        0.0
    )


    traffic_vehicles.append(
        vehicle
    )


    print(
        "Automatic traffic:",
        vehicle.id,
        vehicle.type_id
    )


    # --------------------------------------------------------
    # Maximum traffic
    # --------------------------------------------------------

    if len(traffic_vehicles) >= 25:
        break


print(
    "\nAutomatic traffic vehicles:",
    len(traffic_vehicles)
)

# ============================================================
# COMMON CAMERA PARAMETERS
# ============================================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FOV = 90


# ============================================================
# CAMERA POSITION
# ============================================================

# camera_transform = carla.Transform(
#     carla.Location(
#         x=1.5,
#         z=1.6
#     ),
#     carla.Rotation(
#         pitch=0,
#         yaw=0,
#         roll=0
#     )
# )

camera_transform = carla.Transform(
    carla.Location(
        x=-8.0,       # behind ego vehicle
        y=0.0,
        z=4.0         # above vehicle
    ),
    carla.Rotation(
        pitch=-12.0,  # look slightly downward
        yaw=0.0,
        roll=0.0
    )
)

# ============================================================
# SENSOR STORAGE
# ============================================================

sensors = []


# ============================================================
# SENSOR DATA STORAGE
# ============================================================

rgb_surface = None
depth_raw_surface = None
depth_gray_surface = None
depth_log_surface = None
semantic_surface = None
lidar_surface = None
dvs_surface = None
distorted_surface = None

# LiDAR sensor range (must match lidar_bp "range" attribute below)
LIDAR_MAX_RANGE = 80.0

# ------------------------------------------------------------
# Fixed chase-cam parameters for the in-pygame LiDAR projection.
# The LiDAR points are already expressed in the sensor's local
# frame (they move/rotate WITH the vehicle, same as the RGB image),
# so a constant camera offset here stays rigidly attached to the
# vehicle -- same idea as the RGB camera being attach_to=my_vehicle.
# Coordinates are in the LiDAR's local frame: x forward, y right,
# z up, with the sensor mounted at z=2.5 above the vehicle root.
# ------------------------------------------------------------

LIDAR_CAM_POS = np.array([-9.0, 0.0, 6.0])   # behind and above the sensor
LIDAR_CAM_LOOKAT = np.array([15.0, 0.0, -2.0])  # looking forward and down
LIDAR_CAM_FOCAL = 700.0                       # projection "focal length"


# ============================================================
# RGB CAMERA
# ============================================================

rgb_bp = blueprint_library.find(
    "sensor.camera.rgb"
)

rgb_bp.set_attribute(
    "image_size_x",
    str(IMAGE_WIDTH)
)

rgb_bp.set_attribute(
    "image_size_y",
    str(IMAGE_HEIGHT)
)

rgb_bp.set_attribute(
    "fov",
    str(FOV)
)

rgb_camera = world.spawn_actor(
    rgb_bp,
    camera_transform,
    attach_to=my_vehicle
)

sensors.append(rgb_camera)






# ============================================================
# LIDAR
# ============================================================

lidar_bp = blueprint_library.find(
    "sensor.lidar.ray_cast"
)

lidar_bp.set_attribute(
    "range",
    str(LIDAR_MAX_RANGE)
)

lidar_bp.set_attribute(
    "channels",
    "128"
)

lidar_bp.set_attribute(
    "rotation_frequency",
    "20"
)

lidar_bp.set_attribute(
    "points_per_second",
    "3000000"
)

# CARLA's LiDAR drops a fraction of rays as "noise"/no-return by
# default depending on version -- lowering this keeps more hits.
if lidar_bp.has_attribute("dropoff_general_rate"):
    lidar_bp.set_attribute("dropoff_general_rate", "0.0")

if lidar_bp.has_attribute("dropoff_intensity_limit"):
    lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")

if lidar_bp.has_attribute("dropoff_zero_intensity"):
    lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

lidar_transform = carla.Transform(
    carla.Location(
        x=0,
        z=2.5
    )
)

lidar = world.spawn_actor(
    lidar_bp,
    lidar_transform,
    attach_to=my_vehicle
)

lidar_world_points = []
MAX_ACCUMULATED_POINTS = 1000000

sensors.append(lidar)


# ============================================================
# RGB CALLBACK
# ============================================================

def process_rgb(image):

    global rgb_surface

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    # BGRA -> RGB
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    rgb_surface = pygame.surfarray.make_surface(
        array.swapaxes(0, 1)
    )


rgb_camera.listen(
    process_rgb
)


# ============================================================
# LIDAR TURBO COLORMAP (precomputed 256-entry lookup table)
# ============================================================

_turbo_cmap = cm.get_cmap("turbo")
_TURBO_LUT = (_turbo_cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)

# Rolling buffer of recent (px, py, cam_z, colors) tuples, used to
# stack multiple consecutive scans together for a denser-looking
# cloud. Higher = denser but more motion smear at speed.
LIDAR_HISTORY_FRAMES = 3
_lidar_history = []


# ============================================================
# LIDAR CALLBACK
# Renders a chase-cam perspective projection of the point cloud,
# colored by distance from the sensor (near = warm, far = cool),
# directly into a pygame surface -- same as every other sensor mode,
# no separate Open3D window.
# ============================================================

def process_lidar(point_cloud):

    global lidar_surface

    # This callback runs on every world.tick() no matter which sensor
    # mode is displayed, and synchronous_mode=True means tick() blocks
    # until it returns. Skip the expensive projection/splatting work
    # entirely unless LiDAR is actually the mode being viewed, so
    # switching sensors never silently slows down (and delays) vehicle
    # control response.
    if sensor_mode != 6:
        if _lidar_history:
            _lidar_history.clear()
        return

    # CARLA LiDAR raw format:
    # x, y, z, intensity
    points = np.frombuffer(
        point_cloud.raw_data,
        dtype=np.float32
    ).reshape(-1, 4)[:, :3]

    if points.shape[0] == 0:
        return

    # ----------------------------------------------------
    # Build the camera basis (fixed relative to the vehicle).
    # ----------------------------------------------------

    forward = LIDAR_CAM_LOOKAT - LIDAR_CAM_POS
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    # ----------------------------------------------------
    # Transform points into camera space.
    # ----------------------------------------------------

    relative = points - LIDAR_CAM_POS

    cam_x = relative @ right
    cam_y = relative @ true_up
    cam_z = relative @ forward

    # Keep only points in front of the camera.
    in_front = cam_z > 0.5

    cam_x = cam_x[in_front]
    cam_y = cam_y[in_front]
    cam_z = cam_z[in_front]
    kept_points = points[in_front]

    if cam_x.shape[0] == 0:
        lidar_surface = pygame.Surface((IMAGE_WIDTH, IMAGE_HEIGHT))
        return

    # ----------------------------------------------------
    # Perspective projection to screen space.
    # ----------------------------------------------------

    screen_x = (cam_x / cam_z) * LIDAR_CAM_FOCAL + (IMAGE_WIDTH / 2.0)
    screen_y = -(cam_y / cam_z) * LIDAR_CAM_FOCAL + (IMAGE_HEIGHT / 2.0)

    px = screen_x.astype(np.int32)
    py = screen_y.astype(np.int32)

    on_screen = (
        (px >= 1) & (px < IMAGE_WIDTH - 1) &
        (py >= 1) & (py < IMAGE_HEIGHT - 1)
    )

    px = px[on_screen]
    py = py[on_screen]
    cam_z_vis = cam_z[on_screen]
    kept_points = kept_points[on_screen]

    if px.shape[0] == 0:
        lidar_surface = pygame.Surface((IMAGE_WIDTH, IMAGE_HEIGHT))
        return

    # ----------------------------------------------------
    # Color by distance from the sensor (near = warm, far = cool).
    # ----------------------------------------------------

    distances = np.linalg.norm(kept_points, axis=1)

    normalized_d = np.clip(distances / LIDAR_MAX_RANGE, 0.0, 1.0)

    # Invert so near points are warm and far points are cool.
    lut_indices = (255 * (1.0 - normalized_d)).astype(np.uint8)

    colors = _TURBO_LUT[lut_indices]

    # ----------------------------------------------------
    # Accumulate a short history of frames. Consecutive LiDAR
    # scans are only 0.05s apart, so the vehicle barely moves
    # between them -- stacking a few scans together fills in the
    # gaps between individual laser rings and makes the cloud read
    # as dense as a single higher-resolution scan, at negligible
    # ghosting cost.
    # ----------------------------------------------------

    _lidar_history.append((px, py, cam_z_vis, colors))

    if len(_lidar_history) > LIDAR_HISTORY_FRAMES:
        _lidar_history.pop(0)

    all_px = np.concatenate([f[0] for f in _lidar_history])
    all_py = np.concatenate([f[1] for f in _lidar_history])
    all_cam_z = np.concatenate([f[2] for f in _lidar_history])
    all_colors = np.concatenate([f[3] for f in _lidar_history])

    # ----------------------------------------------------
    # Painter's algorithm: draw far points first, near points
    # last, so closer points correctly occlude farther ones
    # when they land on the same pixel.
    # ----------------------------------------------------

    order = np.argsort(-all_cam_z)

    px = all_px[order]
    py = all_py[order]
    cam_z_vis = all_cam_z[order]
    colors = all_colors[order]

    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    # ----------------------------------------------------
    # Splat each point as a block instead of a single pixel, with
    # the block size growing for closer points (they cover more
    # screen area in a real perspective view too). Tune these
    # radii and distance breakpoints to taste.
    # ----------------------------------------------------

    tier_far = cam_z_vis >= 40.0
    tier_mid = (cam_z_vis >= 15.0) & (cam_z_vis < 40.0)
    tier_near = cam_z_vis < 15.0

    def _splat(mask, offsets):
        if not np.any(mask):
            return
        sub_px = px[mask]
        sub_py = py[mask]
        sub_colors = colors[mask]
        for dx, dy in offsets:
            xs = np.clip(sub_px + dx, 0, IMAGE_WIDTH - 1)
            ys = np.clip(sub_py + dy, 0, IMAGE_HEIGHT - 1)
            image[ys, xs] = sub_colors

    # Far points: 2x2 block.
    _splat(tier_far, ((0, 0), (1, 0), (0, 1), (1, 1)))

    # Mid points: 3x3 block.
    _splat(tier_mid, (
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ))

    # Near points: 5x5 block.
    near_offsets = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    _splat(tier_near, near_offsets)

    lidar_surface = pygame.surfarray.make_surface(
        image.swapaxes(0, 1)
    )


lidar.listen(
    process_lidar
)


# ============================================================
# SENSOR DATA STORAGE GNSS and IMU
# ============================================================

gnss_data = {
    "latitude": 0.0,
    "longitude": 0.0,
    "altitude": 0.0
}


imu_data = {
    "accel_x": 0.0,
    "accel_y": 0.0,
    "accel_z": 0.0,

    "gyro_x": 0.0,
    "gyro_y": 0.0,
    "gyro_z": 0.0,

    "compass": 0.0
}

# ============================================================
# IMU + GYROSCOPE DATA LOGGING
# ============================================================

LOG_DIR = "/home/arif/D_drive/Self_Explore/Autonomous_Vehicle_Stack/AV_Month_works/Month_2/imu_gyro_logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"imu_gyro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

imu_log_file = open(
    log_filename,
    "w",
    newline=""
)

imu_writer = csv.writer(imu_log_file)

imu_writer.writerow([
    "frame",
    "simulation_time",
    
    # Accelerometer
    "accel_x_mps2",
    "accel_y_mps2",
    "accel_z_mps2",

    # Gyroscope
    "gyro_x_radps",
    "gyro_y_radps",
    "gyro_z_radps",

    # Compass
    "compass_rad"
])

print("IMU/Gyroscope recording to:")
print(log_filename)

# ============================================================
# SENSOR CALLBACKS
# ============================================================

def gnss_callback(data):

    gnss_data["latitude"] = data.latitude
    gnss_data["longitude"] = data.longitude
    gnss_data["altitude"] = data.altitude


def imu_callback(data):

    imu_data["accel_x"] = data.accelerometer.x
    imu_data["accel_y"] = data.accelerometer.y
    imu_data["accel_z"] = data.accelerometer.z

    imu_data["gyro_x"] = data.gyroscope.x
    imu_data["gyro_y"] = data.gyroscope.y
    imu_data["gyro_z"] = data.gyroscope.z

    imu_data["compass"] = data.compass


# ============================================================
# GNSS SENSOR
# ============================================================

gnss_bp = blueprint_library.find(
    "sensor.other.gnss"
)

gnss_bp.set_attribute(
    "sensor_tick",
    "0.05"
)


# GNSS mounting position
gnss_transform = carla.Transform(
    carla.Location(
        x=0.0,
        y=0.0,
        z=2.0
    )
)


gnss_sensor = world.spawn_actor(
    gnss_bp,
    gnss_transform,
    attach_to=my_vehicle
)

gnss_sensor.listen(
    gnss_callback
)


print(
    "GNSS sensor attached:",
    gnss_sensor.id
)


# ============================================================
# IMU SENSOR
# ============================================================

imu_bp = blueprint_library.find(
    "sensor.other.imu"
)

imu_bp.set_attribute(
    "sensor_tick",
    "0.05"
)


# IMU mounting position
imu_transform = carla.Transform(
    carla.Location(
        x=0.0,
        y=0.0,
        z=1.5
    )
)


imu_sensor = world.spawn_actor(
    imu_bp,
    imu_transform,
    attach_to=my_vehicle
)

imu_sensor.listen(
    imu_callback
)


print(
    "IMU sensor attached:",
    imu_sensor.id
)


# ============================================================
# PYGAME INITIALIZATION
# ============================================================

pygame.init()

WIDTH = 1200
HEIGHT = 800

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "CARLA GNSS + IMU Visualization"
)

font = pygame.font.SysFont(
    "consolas",
    22
)

small_font = pygame.font.SysFont(
    "consolas",
    18
)

clock = pygame.time.Clock()


# ============================================================
# HISTORY FOR PLOTS
# ============================================================

MAX_POINTS = 300

accel_x_history = []
accel_y_history = []
accel_z_history = []

gyro_x_history = []
gyro_y_history = []
gyro_z_history = []


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    text,
    x,
    y,
    font_object=font
):

    surface = font_object.render(
        text,
        True,
        (255, 255, 255)
    )

    screen.blit(
        surface,
        (x, y)
    )


# ============================================================
# DRAW GRAPH
# ============================================================

def draw_graph(
    values,
    x,
    y,
    width,
    height,
    title
):

    # Background
    pygame.draw.rect(
        screen,
        (25, 25, 25),
        (x, y, width, height)
    )

    pygame.draw.rect(
        screen,
        (100, 100, 100),
        (x, y, width, height),
        1
    )

    draw_text(
        title,
        x + 10,
        y + 5,
        small_font
    )

    if len(values) < 2:
        return

    values = np.array(values)

    # Automatic scale
    max_value = max(
        np.max(np.abs(values)),
        1.0
    )

    points = []

    for i, value in enumerate(values):

        px = (
            x
            + int(
                i
                / (len(values) - 1)
                * width
            )
        )

        normalized = (
            value / max_value
        )

        py = (
            y
            + height / 2
            - normalized
            * height
            / 2
        )

        points.append(
            (px, int(py))
        )

    if len(points) > 1:

        pygame.draw.lines(
            screen,
            (0, 255, 0),
            False,
            points,
            2
        )


# =================================================================
# Initialize pygame
# =================================================================


pygame.init()

screen = pygame.display.set_mode(
    (IMAGE_WIDTH, IMAGE_HEIGHT)
)

pygame.display.set_caption(
    "CARLA - Multi Sensor Viewer"
)

clock = pygame.time.Clock()

font = pygame.font.Font(
    None,
    28
)

small_font = pygame.font.Font(
    None,
    24
)


# ============================================================
# SENSOR MODE
# ============================================================

sensor_mode = 1


sensor_names = {
    1: "RGB Camera",
    2: "Depth - Raw",
    3: "Depth - Grayscale",
    4: "Depth - Logarithmic Grayscale",
    5: "Semantic Segmentation - CityScapes",
    6: "LiDAR",
    7: "Dynamic Vision Sensor",
    8: "RGB Camera - Distorted"
}


# ============================================================
# MANUAL CONTROL
# ============================================================

try:

    running = True

    while running:


        # ====================================================
        # ADVANCE CARLA SIMULATION
        # ====================================================

        world.tick()
        # ====================================================
        # RECORD IMU + GYROSCOPE DATA
        # ====================================================

        snapshot = world.get_snapshot()

        imu_writer.writerow([
            snapshot.frame,
            snapshot.timestamp.elapsed_seconds,

            imu_data["accel_x"],
            imu_data["accel_y"],
            imu_data["accel_z"],

            imu_data["gyro_x"],
            imu_data["gyro_y"],
            imu_data["gyro_z"],

            imu_data["compass"]
        ])

        imu_log_file.flush()

        # ====================================================
        # EVENTS
        # ====================================================



        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                # ESC
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Sensor selection
                elif event.key == pygame.K_1:
                    sensor_mode = 1

                elif event.key == pygame.K_2:
                    sensor_mode = 2

                elif event.key == pygame.K_3:
                    sensor_mode = 3

                elif event.key == pygame.K_4:
                    sensor_mode = 4

                elif event.key == pygame.K_5:
                    sensor_mode = 5

                elif event.key == pygame.K_6:
                    sensor_mode = 6

                elif event.key == pygame.K_7:
                    sensor_mode = 7

                elif event.key == pygame.K_8:
                    sensor_mode = 8


        # ====================================================
        # KEYBOARD
        # ====================================================

        keys = pygame.key.get_pressed()


        # ----------------------------------------------------
        # Throttle
        # ----------------------------------------------------

        if keys[pygame.K_w]:
            throttle = 0.6
        else:
            throttle = 0.0


        # ----------------------------------------------------
        # Brake
        # ----------------------------------------------------

        if keys[pygame.K_s]:
            brake = 1.0
        else:
            brake = 0.0


        # ----------------------------------------------------
        # Steering
        # ----------------------------------------------------

        if keys[pygame.K_a]:
            steer = -0.5

        elif keys[pygame.K_d]:
            steer = 0.5

        else:
            steer = 0.0


        # ----------------------------------------------------
        # Reverse
        # ----------------------------------------------------

        reverse = keys[pygame.K_r]


        # ----------------------------------------------------
        # Hand brake
        # ----------------------------------------------------

        hand_brake = keys[pygame.K_SPACE]


        # ====================================================
        # APPLY VEHICLE CONTROL
        # ====================================================

        control = carla.VehicleControl()

        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        control.reverse = reverse
        control.hand_brake = hand_brake

        my_vehicle.apply_control(
            control
        )


        # ====================================================
        # VEHICLE STATE
        # ====================================================

        velocity = my_vehicle.get_velocity()

        speed = 3.6 * np.sqrt(
            velocity.x ** 2 +
            velocity.y ** 2 +
            velocity.z ** 2
        )


        acceleration = my_vehicle.get_acceleration()

        acceleration_value = np.sqrt(
            acceleration.x ** 2 +
            acceleration.y ** 2 +
            acceleration.z ** 2
        )


        # ============================================================
        # DISPLAY
        # ============================================================

        # ------------------------------------------------------------
        # RGB CAMERA AS MAIN BACKGROUND
        # ------------------------------------------------------------

        if rgb_surface is not None:

            screen.blit(
                rgb_surface,
                (0, 0)
            )

        else:

            screen.fill(
                (0, 0, 0)
            )


        # ============================================================
        # SEMI-TRANSPARENT INFORMATION PANEL
        # ============================================================

        info_panel = pygame.Surface(
            (430, 330),
            pygame.SRCALPHA
        )

        info_panel.fill(
            (0, 0, 0, 170)
        )

        screen.blit(
            info_panel,
            (20, 20)
        )


        # ============================================================
        # TITLE
        # ============================================================

        title = font.render(
            "EGO VEHICLE - LIVE SENSOR DATA",
            True,
            (255, 255, 255)
        )

        screen.blit(
            title,
            (40, 35)
        )


        # ============================================================
        # VEHICLE STATE
        # ============================================================

        draw_text(
            "VEHICLE",
            40,
            80
        )

        draw_text(
            f"Speed        : {speed:7.2f} km/h",
            40,
            115,
            small_font
        )

        draw_text(
            f"Acceleration : {acceleration_value:7.2f} m/s²",
            40,
            145,
            small_font
        )

        draw_text(
            f"Throttle     : {throttle:7.2f}",
            40,
            175,
            small_font
        )

        draw_text(
            f"Brake        : {brake:7.2f}",
            40,
            205,
            small_font
        )

        draw_text(
            f"Steering     : {steer:7.2f}",
            40,
            235,
            small_font
        )


        # ============================================================
        # GNSS
        # ============================================================

        draw_text(
            "GNSS",
            40,
            275
        )

        draw_text(
            f"Lat : {gnss_data['latitude']:.7f}",
            40,
            305,
            small_font
        )

        draw_text(
            f"Lon : {gnss_data['longitude']:.7f}",
            40,
            335,
            small_font
        )

        draw_text(
            f"Alt : {gnss_data['altitude']:.2f} m",
            40,
            365,
            small_font
        )


        # ============================================================
        # IMU PANEL
        # ============================================================

        imu_panel = pygame.Surface(
            (430, 310),
            pygame.SRCALPHA
        )

        imu_panel.fill(
            (0, 0, 0, 170)
        )

        screen.blit(
            imu_panel,
            (830, 20)
        )


        # ============================================================
        # IMU
        # ============================================================

        draw_text(
            "IMU",
            850,
            35
        )


        # ------------------------------------------------------------
        # Accelerometer
        # ------------------------------------------------------------

        draw_text(
            "ACCELEROMETER",
            850,
            80,
            small_font
        )

        draw_text(
            f"X : {imu_data['accel_x']:+8.3f} m/s²",
            850,
            110,
            small_font
        )

        draw_text(
            f"Y : {imu_data['accel_y']:+8.3f} m/s²",
            850,
            140,
            small_font
        )

        draw_text(
            f"Z : {imu_data['accel_z']:+8.3f} m/s²",
            850,
            170,
            small_font
        )


        # ------------------------------------------------------------
        # Gyroscope
        # ------------------------------------------------------------

        draw_text(
            "GYROSCOPE",
            850,
            215,
            small_font
        )

        draw_text(
            f"X : {imu_data['gyro_x']:+8.3f} rad/s",
            850,
            245,
            small_font
        )

        draw_text(
            f"Y : {imu_data['gyro_y']:+8.3f} rad/s",
            850,
            275,
            small_font
        )

        draw_text(
            f"Z : {imu_data['gyro_z']:+8.3f} rad/s",
            850,
            305,
            small_font
        )


        # ============================================================
        # COMPASS
        # ============================================================

        draw_text(
            f"Compass : {imu_data['compass']:.3f} rad",
            850,
            350,
            small_font
        )


        # ============================================================
        # SENSOR STATUS
        # ============================================================

        status_panel = pygame.Surface(
            (430, 120),
            pygame.SRCALPHA
        )

        status_panel.fill(
            (0, 0, 0, 170)
        )

        screen.blit(
            status_panel,
            (830, 400)
        )

        draw_text(
            "SENSORS",
            850,
            415
        )

        draw_text(
            "RGB Camera : ACTIVE",
            850,
            450,
            small_font
        )

        draw_text(
            "GNSS       : ACTIVE",
            850,
            480,
            small_font
        )

        draw_text(
            "IMU        : ACTIVE",
            850,
            510,
            small_font
        )


        # ============================================================
        # CONTROL HELP
        # ============================================================

        help_surface = font.render(
            "W: Accelerate   S: Brake   A/D: Steering   "
            "R: Reverse   SPACE: Handbrake   ESC: Exit",
            True,
            (255, 255, 255)
        )

        screen.blit(
            help_surface,
            (20, IMAGE_HEIGHT - 40)
        )


        # ============================================================
        # UPDATE DISPLAY
        # ============================================================

        pygame.display.flip()

        clock.tick(20)
# ============================================================
# CLEANUP
# ============================================================

finally:

    print("Stopping sensors...")

    for sensor in sensors:
        try:
            sensor.stop()
        except:
            pass

    for sensor in sensors:
        try:
            sensor.destroy()
        except:
            pass

    if my_vehicle is not None:
        try:
            my_vehicle.destroy()
        except:
            pass

    for vehicle in traffic_vehicles:
        try:
            vehicle.destroy()
        except:
            pass

    # Restore Traffic Manager
    try:
        traffic_manager.set_synchronous_mode(False)
    except:
        pass

    # Restore CARLA world
    try:
        world.apply_settings(original_settings)
    except:
        pass


    try:
        gnss_sensor.stop()
        gnss_sensor.destroy()
    except:
        pass


    try:
        imu_sensor.stop()
        imu_sensor.destroy()
    except:
        pass

    # ========================================================
    # STOP SENSORS
    # ========================================================

    try:
        gnss_sensor.stop()
        gnss_sensor.destroy()
    except:
        pass


    try:
        imu_sensor.stop()
        imu_sensor.destroy()
    except:
        pass
    # ========================================================
    # CLOSE IMU/GYROSCOPE LOG FILE
    # ========================================================

    try:
        imu_log_file.close()
        print("IMU/Gyroscope log saved:")
        print(log_filename)
    except:
        pass

    pygame.quit()

    print("Cleanup complete.")