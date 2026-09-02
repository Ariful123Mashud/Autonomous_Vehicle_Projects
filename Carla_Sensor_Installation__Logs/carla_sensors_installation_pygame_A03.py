import carla
import random
import time
import pygame
import numpy as np



# ============================================================
# CARLA CONNECTION
# ============================================================

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)

world = client.load_world("Town01")

time.sleep(3)
world.wait_for_tick()


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

# ============================================================
# SYNCHRONOUS SIMULATION
# ============================================================

settings = world.get_settings()

settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05

world.apply_settings(settings)


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

camera_transform = carla.Transform(
    carla.Location(
        x=1.5,
        z=1.6
    ),
    carla.Rotation(
        pitch=0,
        yaw=0,
        roll=0
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
# DEPTH CAMERA
# ============================================================

depth_bp = blueprint_library.find(
    "sensor.camera.depth"
)

depth_bp.set_attribute(
    "image_size_x",
    str(IMAGE_WIDTH)
)

depth_bp.set_attribute(
    "image_size_y",
    str(IMAGE_HEIGHT)
)

depth_bp.set_attribute(
    "fov",
    str(FOV)
)

depth_camera = world.spawn_actor(
    depth_bp,
    camera_transform,
    attach_to=my_vehicle
)

sensors.append(depth_camera)


# ============================================================
# SEMANTIC SEGMENTATION CAMERA
# ============================================================

semantic_bp = blueprint_library.find(
    "sensor.camera.semantic_segmentation"
)

semantic_bp.set_attribute(
    "image_size_x",
    str(IMAGE_WIDTH)
)

semantic_bp.set_attribute(
    "image_size_y",
    str(IMAGE_HEIGHT)
)

semantic_bp.set_attribute(
    "fov",
    str(FOV)
)

semantic_camera = world.spawn_actor(
    semantic_bp,
    camera_transform,
    attach_to=my_vehicle
)

sensors.append(semantic_camera)


# ============================================================
# LIDAR
# ============================================================

lidar_bp = blueprint_library.find(
    "sensor.lidar.ray_cast"
)

lidar_bp.set_attribute(
    "range",
    "80"
)

lidar_bp.set_attribute(
    "channels",
    "64"
)

lidar_bp.set_attribute(
    "rotation_frequency",
    "20"
)

lidar_bp.set_attribute(
    "points_per_second",
    "1000000"
)

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

sensors.append(lidar)


# ============================================================
# DYNAMIC VISION SENSOR
# ============================================================

dvs_bp = blueprint_library.find(
    "sensor.camera.dvs"
)

dvs_bp.set_attribute(
    "image_size_x",
    str(IMAGE_WIDTH)
)

dvs_bp.set_attribute(
    "image_size_y",
    str(IMAGE_HEIGHT)
)

dvs_bp.set_attribute(
    "fov",
    str(FOV)
)

dvs_camera = world.spawn_actor(
    dvs_bp,
    camera_transform,
    attach_to=my_vehicle
)

sensors.append(dvs_camera)


# ============================================================
# DISTORTED RGB CAMERA
# ============================================================

distorted_bp = blueprint_library.find(
    "sensor.camera.rgb"
)

distorted_bp.set_attribute(
    "image_size_x",
    str(IMAGE_WIDTH)
)

distorted_bp.set_attribute(
    "image_size_y",
    str(IMAGE_HEIGHT)
)

distorted_bp.set_attribute(
    "fov",
    str(FOV)
)

# Lens distortion parameters
distorted_bp.set_attribute(
    "lens_circle_multiplier",
    "1.0"
)

distorted_bp.set_attribute(
    "lens_circle_falloff",
    "5.0"
)

distorted_bp.set_attribute(
    "lens_k",
    "0.1"
)

distorted_bp.set_attribute(
    "lens_kcube",
    "0.0"
)

distorted_bp.set_attribute(
    "lens_x_size",
    "0.08"
)

distorted_bp.set_attribute(
    "lens_y_size",
    "0.08"
)

distorted_camera = world.spawn_actor(
    distorted_bp,
    camera_transform,
    attach_to=my_vehicle
)

sensors.append(distorted_camera)


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
# DEPTH CALLBACK
# ============================================================

def process_depth(image):

    global depth_raw_surface
    global depth_gray_surface
    global depth_log_surface

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    # --------------------------------------------------------
    # CARLA depth encoding
    # --------------------------------------------------------

    depth = (
        array[:, :, 2].astype(np.float32)
        +
        array[:, :, 1].astype(np.float32) * 256.0
        +
        array[:, :, 0].astype(np.float32) * 256.0 * 256.0
    )

    depth = depth / (256.0 ** 3 - 1)

    # Depth in meters
    depth_meters = depth * 1000.0


    # --------------------------------------------------------
    # RAW DEPTH
    #
    # Display actual depth after clipping to 80 m.
    # --------------------------------------------------------

    raw_display = np.clip(
        depth_meters / 80.0 * 255.0,
        0,
        255
    ).astype(np.uint8)

    raw_rgb = np.stack(
        [raw_display] * 3,
        axis=2
    )

    depth_raw_surface = pygame.surfarray.make_surface(
        raw_rgb.swapaxes(0, 1)
    )


    # --------------------------------------------------------
    # LINEAR GRAYSCALE DEPTH
    # --------------------------------------------------------

    gray_display = np.clip(
        depth_meters / 80.0 * 255.0,
        0,
        255
    ).astype(np.uint8)

    gray_rgb = np.stack(
        [gray_display] * 3,
        axis=2
    )

    depth_gray_surface = pygame.surfarray.make_surface(
        gray_rgb.swapaxes(0, 1)
    )


    # --------------------------------------------------------
    # LOGARITHMIC DEPTH
    # --------------------------------------------------------

    log_depth = np.log1p(
        depth_meters
    )

    log_max = np.log1p(80.0)

    log_display = np.clip(
        log_depth / log_max * 255.0,
        0,
        255
    ).astype(np.uint8)

    log_rgb = np.stack(
        [log_display] * 3,
        axis=2
    )

    depth_log_surface = pygame.surfarray.make_surface(
        log_rgb.swapaxes(0, 1)
    )


depth_camera.listen(
    process_depth
)


# ============================================================
# SEMANTIC SEGMENTATION CALLBACK
# ============================================================

def process_semantic(image):

    global semantic_surface

    # Convert to CityScapes palette
    image.convert(
        carla.ColorConverter.CityScapesPalette
    )

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    array = array[:, :, :3]

    # BGRA/BGR -> RGB
    array = array[:, :, ::-1]

    semantic_surface = pygame.surfarray.make_surface(
        array.swapaxes(0, 1)
    )


semantic_camera.listen(
    process_semantic
)


# ============================================================
# LIDAR CALLBACK
# ============================================================

def process_lidar(point_cloud):

    global lidar_surface

    points = np.frombuffer(
        point_cloud.raw_data,
        dtype=np.float32
    )

    points = np.reshape(
        points,
        (-1, 4)
    )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # --------------------------------------------------------
    # Create black image for LiDAR
    # --------------------------------------------------------

    lidar_image = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Project top-view LiDAR points
    # --------------------------------------------------------

    lidar_range = 50.0

    valid = (
        (x > -lidar_range) &
        (x < lidar_range) &
        (y > -lidar_range) &
        (y < lidar_range)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]


    # Convert world coordinates to image coordinates
    px = (
        (y + lidar_range)
        / (2 * lidar_range)
        * (IMAGE_WIDTH - 1)
    ).astype(np.int32)

    py = (
        (x + lidar_range)
        / (2 * lidar_range)
        * (IMAGE_HEIGHT - 1)
    ).astype(np.int32)

    valid_pixels = (
        (px >= 0) &
        (px < IMAGE_WIDTH) &
        (py >= 0) &
        (py < IMAGE_HEIGHT)
    )

    px = px[valid_pixels]
    py = py[valid_pixels]

    # Draw LiDAR points
    lidar_image[
        py,
        px
    ] = 255

    lidar_surface = pygame.surfarray.make_surface(
        lidar_image.swapaxes(0, 1)
    )


lidar.listen(
    process_lidar
)


# ============================================================
# DVS CALLBACK
# ============================================================

def process_dvs(events):

    global dvs_surface

    # Create black image
    dvs_image = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8
    )

    if len(events) == 0:

        dvs_surface = pygame.surfarray.make_surface(
            dvs_image.swapaxes(0, 1)
        )

        return


    # ========================================================
    # DVS EVENT DATA
    # ========================================================

    x = np.array(
        [event.x for event in events],
        dtype=np.int32
    )

    y = np.array(
        [event.y for event in events],
        dtype=np.int32
    )

    # IMPORTANT:
    # CARLA DVSEvent uses "pol", not "polarity"
    polarity = np.array(
        [event.pol for event in events],
        dtype=np.int8
    )


    # ========================================================
    # VALID PIXELS
    # ========================================================

    valid = (
        (x >= 0) &
        (x < IMAGE_WIDTH) &
        (y >= 0) &
        (y < IMAGE_HEIGHT)
    )

    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]


    # ========================================================
    # POSITIVE EVENTS
    # ========================================================

    positive = polarity > 0

    dvs_image[
        y[positive],
        x[positive]
    ] = [255, 255, 255]


    # ========================================================
    # NEGATIVE EVENTS
    # ========================================================

    negative = polarity <= 0

    dvs_image[
        y[negative],
        x[negative]
    ] = [100, 100, 100]


    # ========================================================
    # PYGAME SURFACE
    # ========================================================

    dvs_surface = pygame.surfarray.make_surface(
        dvs_image.swapaxes(0, 1)
    )


dvs_camera.listen(
    process_dvs
)


# ============================================================
# DISTORTED RGB CALLBACK
# ============================================================

def process_distorted(image):

    global distorted_surface

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    array = array[:, :, :3]

    # BGR -> RGB
    array = array[:, :, ::-1]

    distorted_surface = pygame.surfarray.make_surface(
        array.swapaxes(0, 1)
    )


distorted_camera.listen(
    process_distorted
)


# ============================================================
# PYGAME INITIALIZATION
# ============================================================

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


        # ====================================================
        # SELECT SENSOR IMAGE
        # ====================================================

        current_surface = None

        if sensor_mode == 1:
            current_surface = rgb_surface

        elif sensor_mode == 2:
            current_surface = depth_raw_surface

        elif sensor_mode == 3:
            current_surface = depth_gray_surface

        elif sensor_mode == 4:
            current_surface = depth_log_surface

        elif sensor_mode == 5:
            current_surface = semantic_surface

        elif sensor_mode == 6:
            current_surface = lidar_surface

        elif sensor_mode == 7:
            current_surface = dvs_surface

        elif sensor_mode == 8:
            current_surface = distorted_surface


        # ====================================================
        # DISPLAY SENSOR
        # ====================================================

        if current_surface is not None:

            screen.blit(
                current_surface,
                (0, 0)
            )

        else:

            screen.fill(
                (0, 0, 0)
            )


        # ====================================================
        # SENSOR NAME
        # ====================================================

        sensor_text = font.render(
            f"[{sensor_mode}] {sensor_names[sensor_mode]}",
            True,
            (255, 255, 255)
        )

        screen.blit(
            sensor_text,
            (20, 20)
        )


        # ====================================================
        # VEHICLE INFORMATION
        # ====================================================

        info = [
            f"Throttle:      {throttle:.2f}",
            f"Brake:         {brake:.2f}",
            f"Steering:      {steer:.2f}",
            f"Speed:         {speed:.1f} km/h",
            f"Acceleration:  {acceleration_value:.2f} m/s2",
            f"Reverse:       {reverse}",
            f"Hand Brake:    {hand_brake}"
        ]


        y = 60

        for text in info:

            text_surface = small_font.render(
                text,
                True,
                (255, 255, 255)
            )

            screen.blit(
                text_surface,
                (20, y)
            )

            y += 27


        # ====================================================
        # HELP
        # ====================================================

        help_text = small_font.render(
            "1 RGB | 2 Depth | 3 Gray | 4 Log | "
            "5 Semantic | 6 LiDAR | 7 DVS | 8 Distorted",
            True,
            (255, 255, 255)
        )

        screen.blit(
            help_text,
            (20, IMAGE_HEIGHT - 35)
        )


        # ====================================================
        # UPDATE DISPLAY
        # ====================================================

        pygame.display.flip()

        clock.tick(60)


# ============================================================
# CLEANUP
# ============================================================

finally:

    print("Stopping sensors...")

    # Stop sensor listeners
    for sensor in sensors:

        try:
            sensor.stop()
        except RuntimeError:
            pass


    # Destroy sensors
    for sensor in sensors:

        try:
            sensor.destroy()
        except RuntimeError:
            pass


    # Destroy ego vehicle
    if my_vehicle is not None:

        try:
            my_vehicle.destroy()
        except RuntimeError:
            pass


    # Destroy traffic
    for vehicle in traffic_vehicles:

        try:
            vehicle.destroy()
        except RuntimeError:
            pass


    pygame.quit()

    print("Cleanup complete.")