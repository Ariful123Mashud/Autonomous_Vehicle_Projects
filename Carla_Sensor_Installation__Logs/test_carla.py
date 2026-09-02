import carla
import time

print("Connecting to CARLA...")

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

print("Connected")

world = client.get_world()

print("World:", world.get_map().name)

vehicles = world.get_actors().filter("vehicle.*")

print("Vehicles:", len(vehicles))

print("CARLA connection successful")

time.sleep(10)