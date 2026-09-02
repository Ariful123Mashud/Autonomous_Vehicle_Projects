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

time.sleep(2)


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

traffic_manager.set_global_distance_to_leading_vehicle(3.0)


# ============================================================
# SPAWN POINTS
# ============================================================

spawn_points = world.get_map().get_spawn_points()


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
    raise RuntimeError("Failed to spawn ego vehicle")

print("Ego vehicle:", my_vehicle.type_id)

my_vehicle.set_autopilot(False)


# ============================================================
# TRAFFIC
# ============================================================

traffic_vehicles = []

traffic_blueprints = blueprint_library.filter(
    "vehicle.*"
)

for i in range(20):

    vehicle_bp = random.choice(
        traffic_blueprints
    )

    spawn_point = random.choice(
        spawn_points
    )

    # Keep traffic away from ego spawn point
    if spawn_point.location.distance(
        my_spawn_point.location
    ) < 8.0:
        continue

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
    )

    if vehicle is not None:

        vehicle.set_autopilot(
            True,
            traffic_manager.get_port()
        )

        traffic_vehicles.append(vehicle)


print(
    "Traffic vehicles:",
    len(traffic_vehicles)
)


# ============================================================
# RGB CAMERA
# ============================================================

camera_bp = blueprint_library.find(
    "sensor.camera.rgb"
)

# Image resolution
camera_bp.set_attribute(
    "image_size_x",
    "1280"
)

camera_bp.set_attribute(
    "image_size_y",
    "720"
)

camera_bp.set_attribute(
    "fov",
    "90"
)


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
# SPAWN CAMERA
# ============================================================

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=my_vehicle
)


print("RGB camera attached.")


# ============================================================
# CAMERA IMAGE STORAGE
# ============================================================

camera_surface = None


def process_image(image):

    global camera_surface

    # Convert CARLA image to numpy
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    # CARLA uses BGRA
    array = array[:, :, :3]

    # Convert BGR -> RGB
    array = array[:, :, ::-1]

    # Convert to pygame surface
    camera_surface = pygame.surfarray.make_surface(
        array.swapaxes(0, 1)
    )


camera.listen(
    lambda image: process_image(image)
)


# ============================================================
# PYGAME
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (1280, 720)
)

pygame.display.set_caption(
    "CARLA - Manual Vehicle Control"
)

clock = pygame.time.Clock()


# ============================================================
# MANUAL CONTROL
# ============================================================

try:

    running = True

    while running:

        # ----------------------------------------------------
        # Events
        # ----------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False


        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        keys = pygame.key.get_pressed()


        # Throttle
        if keys[pygame.K_w]:
            throttle = 0.6
        else:
            throttle = 0.0


        # Brake
        if keys[pygame.K_s]:
            brake = 1.0
        else:
            brake = 0.0


        # Steering
        if keys[pygame.K_a]:
            steer = -0.5

        elif keys[pygame.K_d]:
            steer = 0.5

        else:
            steer = 0.0


        # Reverse
        reverse = keys[pygame.K_r]


        # Hand brake
        hand_brake = keys[pygame.K_SPACE]


        # ----------------------------------------------------
        # Apply CARLA control
        # ----------------------------------------------------

        control = carla.VehicleControl()

        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        control.reverse = reverse
        control.hand_brake = hand_brake

        my_vehicle.apply_control(control)


        # ----------------------------------------------------
        # Display camera image
        # ----------------------------------------------------

        if camera_surface is not None:

            screen.blit(
                camera_surface,
                (0, 0)
            )

        else:

            screen.fill(
                (0, 0, 0)
            )


        pygame.display.flip()

        clock.tick(60)


# ============================================================
# CLEANUP
# ============================================================

finally:

    print("Stopping...")

    camera.stop()
    camera.destroy()

    if my_vehicle is not None:
        try:
            my_vehicle.destroy()
        except RuntimeError:
            pass

    for vehicle in traffic_vehicles:

        try:
            vehicle.destroy()
        except RuntimeError:
            pass

    pygame.quit()

    print("Cleanup complete.")