# Sound Source locate
# 
# @Time    : 2019-12-04 14:44
# @Author  : xyzhao
# @File    : markov.py
# @Description:

import cv2
import numpy as np
import random

# Things to try:
# Rolling average probabilities
# Change number particles over time
# Change exploration amount over time
# Snarter averaging
#    - Clustring
#    - Weighted average
# Things attempted:
# - Exploration based on number of zero probability particles
#   Didn't stay converged
# - Added distance difference cut off for zeroing probabilities
#   Worked
#   Making the difference cut off based noise level was iffy
#   Constant noise worked better


# World coordinates are scaled from 0 to 1
config = {
    "image_width": 800,
    "image_height": 800,
    "number_particles": 3000,
    "odometry_noise_mean": 0,  # Bias on longitudinal movement readings (-1 to 1)
    "odometry_noise_std": 0.003,  # Std deviation on longitudinal movement readings (0 to 1)
    "heading_noise_mean": 0,  # Bias on heading movement readings (-pi to pi)
    "heading_noise_std": np.pi / 180,  # Std deviation on heading movement readings (0 to 2pi)
    "ranger_noise_mean": 0,  # Bias on ranger measurements (-1 to 1)
    "ranger_noise_std_prop": 0.03,
    # Scaler that makes std deviation proportional to distance on range measurements (0 to 1)
    "ranger_max_dist": 0.8  # 0 to 1
}


class Robot():
    def __init__(self, config, world):
        self.config = config
        self.world = world

        # Init robot state
        self.pos = np.array((0.2, 0.2, np.pi))
        self.pos = np.random.uniform((0.2, 0.2, 0), (0.6, 0.6, 2 * np.pi), (1, 3))[0]
        self.ranger = 0.8

    def update(self):
        # Run an avoid wall controller
        commanded_movement = self._avoid_walls(self.ranger)
        # Add noise to commanded movement because we want to simulate a noisy controller
        # This does not directly affect the performance of the particle filter, it just makes the sim more relalistic
        # but is actaully makes measurements more unique which actually does help
        bias = np.array((0, 0), dtype=np.float)  # (odom_bias, heading_bias)
        stddev = np.array((0.0005, np.pi / 40), dtype=np.float)  # (odom_std, heading_std)
        applied_movement = np.random.normal(commanded_movement + bias, stddev, (1, 2))[0]
        self._apply_movement(applied_movement)

        # Add noise to the measured movement before we return it because
        # we asssume measured values have noise. These noise characteristics match the ones we model in the pf
        bias = np.array((self.config["odometry_noise_mean"], self.config["heading_noise_mean"]),
                        dtype=np.float)  # (odom_bias, heading_bias)
        stddev = np.array((self.config["odometry_noise_std"], self.config["heading_noise_std"]),
                          dtype=np.float)  # (odom_std, heading_std)
        measured_movement = np.random.normal(applied_movement + bias, stddev, (1, 2))[0]

        # Measure distance
        self.ranger = self.world.distance_to_closest_wall(self.pos)
        # Apply noise - matches noise modelled in pf
        bias = self.config["ranger_noise_mean"]
        stddev = self.config["ranger_noise_std_prop"] * self.ranger
        self.ranger += np.random.normal(bias, stddev, 1)[0]

        return measured_movement, self.ranger

    def _apply_movement(self, movement):
        self.pos[2] += movement[1]  # Move heading
        self.pos[0] += movement[0] * np.cos(self.pos[2])  # Move x based on heading and step size
        self.pos[1] += movement[0] * np.sin(self.pos[2])  # Move y based on heading and step size

    def _avoid_walls(self, ranger):
        movement = np.array((0.005, (0.2 - min(ranger, 0.2)) * -1.2))
        return movement


class ParticleFilter():
    def __init__(self, config, world):
        # Grab map
        self.world = world
        self.config = config

        # Create n particles uniformly sampled over state space (1x1 grid) with the same probability
        state_vector_size = (self.config["number_particles"], 4)  # x+y+heading+probability
        self.particles = np.random.uniform((0, 0.1, 0, 1), (1, 0.7, 2 * np.pi, 1), state_vector_size)

    def update(self, robot_movement, robot_ranger_measurement):
        self.apply_movement(robot_movement)
        self.calculate_particle_probability(robot_ranger_measurement)
        self.resample()

    def apply_movement(self, robot_movement):
        # Define gaussian noise for movement
        bias = np.array((self.config["odometry_noise_mean"], self.config["heading_noise_mean"]),
                        dtype=np.float)  # (odom_bias, heading_bias)
        stddev = np.array((self.config["odometry_noise_std"], self.config["heading_noise_std"]),
                          dtype=np.float)  # (odom_std, heading_std)

        # Apply noise to movement
        movement_with_noise = np.random.normal(robot_movement + bias, stddev, (self.config["number_particles"], 2))

        # Apply noisy movement to particles
        self.particles[:, 2] += movement_with_noise[:, 1]  # Move heading
        self.particles[:, 0] += movement_with_noise[:, 0] * np.cos(
            self.particles[:, 2])  # Move x based on heading and step size
        self.particles[:, 1] += movement_with_noise[:, 0] * np.sin(
            self.particles[:, 2])  # Move y based on heading and step size

    def calculate_particle_probability(self, robot_ranger_measurement):
        particle_ranger_measurements = np.zeros(self.config["number_particles"], np.float64)
        for i, particle in enumerate(self.particles):
            # Simulate ranger for every particle
            particle_ranger_measurements[i] = world.distance_to_closest_wall(particle[:3])
            # Apply noise
            bias = self.config["ranger_noise_mean"]
            stddev = self.config["ranger_noise_std_prop"] * particle_ranger_measurements[i] * 2
            particle_ranger_measurements[i] += np.random.normal(bias, stddev, 1)[0]

        # Calculate similarity between robot and particles
        # Take the difference between robot and particle rangers
        # then exponentially scale it then invert it then threshhold it to become a probability
        diff = (robot_ranger_measurement - particle_ranger_measurements) ** 2
        inv_normalize_diff = 1 - diff / np.max(diff)
        # If the error is large enough set the probability to zero
        # thresh = self.config["ranger_noise_std_prop"] * robot_ranger_measurement * 10
        thresh = 0.1 ** 2
        inv_normalize_diff[diff > thresh] = 0
        # Make probabilities add to 1
        probabilities = inv_normalize_diff / np.sum(inv_normalize_diff)
        # Store
        self.particles[:, 3] = probabilities

    def resample(self):
        probabilities = self.particles[:, 3]
        idx = np.random.choice(self.config["number_particles"], self.config["number_particles"], p=probabilities)
        self.particles = self.particles[idx]
        # explore - uniform resample some particles from state space so we dont get stuck
        # explore_particles = np.count_nonzero(probabilities==0)/10
        # print(explore_particles)
        explore_particles = 20
        self.particles[:explore_particles] = np.random.uniform((0, 0.1, 0, 0), (1, 0.7, 2 * np.pi, 0),
                                                               (explore_particles, 4))


class World():
    walls = np.array([
        [[0.03333333, 0.13333333], [0.03333333, 0.26666667]],
        [[0.03333333, 0.26666667], [0.16666667, 0.7]],
        [[0.16666667, 0.7], [0.26666667, 0.7]],
        [[0.26666667, 0.7], [0.4, 0.6]],
        [[0.4, 0.6], [0.66666667, 0.6]],
        [[0.66666667, 0.6], [0.66666667, 0.53333333]],
        [[0.66666667, 0.53333333], [0.83333333, 0.43333333]],
        [[0.83333333, 0.43333333], [0.83333333, 0.26666667]],
        [[0.83333333, 0.26666667], [1., 0.26666667]],
        [[1., 0.26666667], [1., 0.1]],
        [[1., 0.1], [0.33333333, 0.1]],
        [[0.33333333, 0.1], [0.33333333, 0.13333333]],
        [[0.33333333, 0.13333333], [0.03333333, 0.13333333]]
    ], dtype=np.float64)

    def __init__(self, config):
        self.config = config

    def distance_to_closest_wall(self, pos):
        x, y, heading = pos
        ranger_endpoint_x = x + self.config["ranger_max_dist"] * np.cos(heading)
        ranger_endpoint_y = y + self.config["ranger_max_dist"] * np.sin(heading)
        ranger_segment = np.array([[x, y], [ranger_endpoint_x, ranger_endpoint_y]])

        min_dist = np.inf
        for wall in self.walls:
            intersection = self._segment_intersection(wall, ranger_segment)
            if intersection is not None:
                dist = np.sqrt((x - intersection[0]) ** 2 + (y - intersection[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist

        if min_dist == np.inf:
            min_dist = self.config["ranger_max_dist"]

        return min_dist

    def _segment_intersection(self, segA, segB):
        (pt_a0x, pt_a0y), (pt_a1x, pt_a1y) = segA
        (pt_b0x, pt_b0y), (pt_b1x, pt_b1y) = segB

        s1_x = pt_a1x - pt_a0x
        s1_y = pt_a1y - pt_a0y
        s2_x = pt_b1x - pt_b0x
        s2_y = pt_b1y - pt_b0y

        s = (-s1_y * (pt_a0x - pt_b0x) + s1_x * (pt_a0y - pt_b0y)) / (-s2_x * s1_y + s1_x * s2_y)
        t = (s2_x * (pt_a0y - pt_b0y) - s2_y * (pt_a0x - pt_b0x)) / (-s2_x * s1_y + s1_x * s2_y)

        if 0 <= s <= 1 and 0 <= t <= 1:
            # Collision detected
            i_x = pt_a0x + (t * s1_x)
            i_y = pt_a0y + (t * s1_y)
            return np.array([i_x, i_y])

        return None


class Viz():
    def __init__(self, config):
        self.config = config
        self.unit2pix = self.config["image_width"]

    def draw(self, world, robot, particles):
        image = np.ones((config["image_height"], config["image_width"], 3), dtype=np.uint8) * 240

        image = self.draw_map(image, world)
        image = self.draw_robot(image, robot)
        image = self.draw_particles(image, particles)

        print("hello")
        cv2.imshow("Robot", image)
        cv2.waitKey(1)
        print("hello2")

    def world_to_image(self, pnt):
        xi = int(pnt[0] * self.unit2pix)
        yi = int(self.config["image_height"] - (pnt[1] * self.unit2pix))
        return xi, yi

    def draw_robot(self, image, robot):
        x0, y0, heading = robot
        x0i, y0i = self.world_to_image((x0, y0))
        color = (255, 34, 4)
        # Robot
        cv2.circle(image, (x0i, y0i), 6, color, 1, -1)
        # Ranger
        x1 = self.config["ranger_max_dist"] * np.cos(heading) + x0
        y1 = self.config["ranger_max_dist"] * np.sin(heading) + y0
        x1i, y1i = self.world_to_image((x1, y1))
        cv2.line(image, (x0i, y0i), (x1i, y1i), color, 1)
        return image

    def draw_map(self, image, world):
        lineThickness = 2
        for wall in world.walls:
            (x0, y0), (x1, y1) = wall
            x0 = int(x0 * self.unit2pix)
            y0 = int(self.config["image_height"] - (y0 * self.unit2pix))
            x1 = int(x1 * self.unit2pix)
            y1 = int(self.config["image_height"] - (y1 * self.unit2pix))
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), lineThickness)
        return image

    def draw_particles(self, image, particles):
        for particle in particles:
            x, y, heading, prob = particle
            x = int(x * self.unit2pix)
            y = int(self.config["image_height"] - (y * self.unit2pix))
            color = int(prob * 255)
            cv2.circle(image, (x, y), 1, color, -1, -1)

        x = np.mean(particles[:, 0])
        y = np.mean(particles[:, 1])
        cv2.circle(image, self.world_to_image((x, y)), 3, (0, 0, 255), 1, -1)
        return image


if __name__ == "__main__":

    world = World(config)
    viz = Viz(config)
    pf = ParticleFilter(config, world)
    robot = Robot(config, world)

    while True:
        print("hh")
        movement, ranger = robot.update()
        pf.update(movement, ranger)
        viz.draw(world, robot.pos, pf.particles)
