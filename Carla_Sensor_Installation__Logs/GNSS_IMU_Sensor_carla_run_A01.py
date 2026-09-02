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

world = client.get_world()

original_settings = world.get_settings()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05

world.apply_settings(settings)

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)

print("Synchronous simulation enabled")


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

traffic_manager.set_synchronous_mode(True)

traffic_manager.set_global_distance_to_leading_vehicle(
    8.0
)

traffic_manager.global_percentage_speed_difference(
    30.0
)


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

# Manual vehicle
my_vehicle.set_autopilot(False)


# ============================================================
# TRAFFIC
# ============================================================

traffic_vehicles = []

traffic_blueprints = blueprint_library.filter(
    "vehicle.*"
)

available_spawn_points = spawn_points.copy()

random.shuffle(
    available_spawn_points
)


for spawn_point in available_spawn_points:

    # Don't spawn too close to ego
    if spawn_point.location.distance(
        my_spawn_point.location
    ) < 30.0:

        continue

    vehicle_bp = random.choice(
        traffic_blueprints
    )

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
    )

    if vehicle is None:
        continue


    # ========================================================
    # TRAFFIC MANAGER
    # ========================================================

    vehicle.set_autopilot(
        True,
        traffic_manager.get_port()
    )

    traffic_manager.distance_to_leading_vehicle(
        vehicle,
        8.0
    )

    traffic_manager.vehicle_percentage_speed_difference(
        vehicle,
        30.0
    )

    traffic_manager.auto_lane_change(
        vehicle,
        False
    )

    traffic_manager.random_left_lanechange_percentage(
        vehicle,
        0.0
    )

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

    if len(traffic_vehicles) >= 25:
        break


print(
    "\nAutomatic traffic vehicles:",
    len(traffic_vehicles)
)


# ============================================================
# SENSOR DATA STORAGE
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


# ============================================================
# MAIN LOOP
# ============================================================

running = True

try:

    while running:

        # ----------------------------------------------------
        # PYGAME EVENTS
        # ----------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    running = False


        # ----------------------------------------------------
        # ADVANCE CARLA
        # ----------------------------------------------------

        world.tick()


        # ----------------------------------------------------
        # STORE IMU HISTORY
        # ----------------------------------------------------

        accel_x_history.append(
            imu_data["accel_x"]
        )

        accel_y_history.append(
            imu_data["accel_y"]
        )

        accel_z_history.append(
            imu_data["accel_z"]
        )

        gyro_x_history.append(
            imu_data["gyro_x"]
        )

        gyro_y_history.append(
            imu_data["gyro_y"]
        )

        gyro_z_history.append(
            imu_data["gyro_z"]
        )


        # Keep only last MAX_POINTS
        accel_x_history = (
            accel_x_history[-MAX_POINTS:]
        )

        accel_y_history = (
            accel_y_history[-MAX_POINTS:]
        )

        accel_z_history = (
            accel_z_history[-MAX_POINTS:]
        )

        gyro_x_history = (
            gyro_x_history[-MAX_POINTS:]
        )

        gyro_y_history = (
            gyro_y_history[-MAX_POINTS:]
        )

        gyro_z_history = (
            gyro_z_history[-MAX_POINTS:]
        )


        # ----------------------------------------------------
        # CLEAR SCREEN
        # ----------------------------------------------------

        screen.fill(
            (15, 15, 15)
        )


        # ====================================================
        # TITLE
        # ====================================================

        draw_text(
            "CARLA EGO VEHICLE SENSOR DATA",
            30,
            20
        )


        # ====================================================
        # GNSS DATA
        # ====================================================

        draw_text(
            "GNSS",
            30,
            75
        )

        draw_text(
            f"Latitude  : "
            f"{gnss_data['latitude']:.8f}",
            30,
            115,
            small_font
        )

        draw_text(
            f"Longitude : "
            f"{gnss_data['longitude']:.8f}",
            30,
            145,
            small_font
        )

        draw_text(
            f"Altitude  : "
            f"{gnss_data['altitude']:.3f} m",
            30,
            175,
            small_font
        )


        # ====================================================
        # IMU DATA
        # ====================================================

        draw_text(
            "IMU",
            400,
            75
        )


        draw_text(
            f"Acceleration X : "
            f"{imu_data['accel_x']:+8.3f} m/s²",
            400,
            115,
            small_font
        )

        draw_text(
            f"Acceleration Y : "
            f"{imu_data['accel_y']:+8.3f} m/s²",
            400,
            145,
            small_font
        )

        draw_text(
            f"Acceleration Z : "
            f"{imu_data['accel_z']:+8.3f} m/s²",
            400,
            175,
            small_font
        )


        draw_text(
            f"Gyroscope X : "
            f"{imu_data['gyro_x']:+8.3f} rad/s",
            400,
            220,
            small_font
        )

        draw_text(
            f"Gyroscope Y : "
            f"{imu_data['gyro_y']:+8.3f} rad/s",
            400,
            250,
            small_font
        )

        draw_text(
            f"Gyroscope Z : "
            f"{imu_data['gyro_z']:+8.3f} rad/s",
            400,
            280,
            small_font
        )


        draw_text(
            f"Compass : "
            f"{imu_data['compass']:.3f} rad",
            400,
            325,
            small_font
        )


        # ====================================================
        # ACCELERATION GRAPH
        # ====================================================

        draw_graph(
            accel_x_history,
            30,
            380,
            350,
            180,
            "Acceleration X"
        )

        draw_graph(
            accel_y_history,
            425,
            380,
            350,
            180,
            "Acceleration Y"
        )

        draw_graph(
            accel_z_history,
            820,
            380,
            350,
            180,
            "Acceleration Z"
        )


        # ====================================================
        # GYROSCOPE GRAPH
        # ====================================================

        draw_graph(
            gyro_x_history,
            30,
            590,
            350,
            180,
            "Gyroscope X"
        )

        draw_graph(
            gyro_y_history,
            425,
            590,
            350,
            180,
            "Gyroscope Y"
        )

        draw_graph(
            gyro_z_history,
            820,
            590,
            350,
            180,
            "Gyroscope Z"
        )


        # ====================================================
        # UPDATE DISPLAY
        # ====================================================

        pygame.display.flip()

        clock.tick(60)


finally:

    print("\nCleaning up...")

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
    # DESTROY EGO VEHICLE
    # ========================================================

    try:
        my_vehicle.destroy()
    except:
        pass


    # ========================================================
    # DESTROY TRAFFIC
    # ========================================================

    for vehicle in traffic_vehicles:

        try:
            vehicle.destroy()
        except:
            pass


    # ========================================================
    # RESTORE CARLA SETTINGS
    # ========================================================

    world.apply_settings(
        original_settings
    )

    traffic_manager.set_synchronous_mode(
        False
    )

    pygame.quit()

    print(
        "Simulation cleaned up."
    )