import carla
import time

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

settings = world.get_settings()

settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05

world.apply_settings(settings)

print("Synchronous mode enabled")

for i in range(200):
    world.tick()

print("200 ticks completed")

settings.synchronous_mode = False
world.apply_settings(settings)

print("Done")