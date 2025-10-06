import math
import PolygonCollision
import cv2
import pygame
from enum import Enum
import numpy as np

pygame.init()
class Color(Enum):
    RED = (255,0,0)
    GREEN = (0,255,0)
    BLUE = (0,0,255)
def get_rect_vertices(cx, cy, w, h, angle_deg):
    theta = math.radians(angle_deg)
    dx = w / 2
    dy = h / 2
    local_corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    vertices = []
    for x, y in local_corners:
        x_rot = x * math.cos(theta) - y * math.sin(theta)
        y_rot = x * math.sin(theta) + y * math.cos(theta)
        vertices.append((cx + x_rot, cy + y_rot))
    return vertices
image_map = cv2.imread('map.jpg',cv2.IMREAD_GRAYSCALE)
_,binary_image = cv2.threshold(image_map,127,255,cv2.THRESH_BINARY)
binary_matrix = (binary_image == 255).astype(int)
binary_matrix = np.array(binary_matrix, dtype=np.uint8)
block_size = 4
map_width = binary_matrix.shape[1]
map_height = binary_matrix.shape[0]
Starting_location = (map_width*block_size // 2, map_height*block_size - block_size*30)
map_surface = pygame.Surface((map_width*block_size,map_height*block_size))
font = pygame.font.Font(None, 45)
for i in range(binary_matrix.shape[1]):
    for j in range(binary_matrix.shape[0]):
        pygame.draw.rect(map_surface, ((255, 255, 255) if binary_matrix[j][i] == 1 else (0, 0, 0)),
                         [i * block_size, j * block_size, block_size, block_size])
class Car:
    def __init__(self):
        self.x = Starting_location[0]
        self.y = Starting_location[1]
        self.speed = 0
        self.max_speed = 5
        self.direction = 0
        self.rotationspeed = 20
        self.horizontal_velocity = 0
        self.vertical_velocity = 0
        self.width = block_size *5
        self.height = block_size *4
        self.alive = True
        self.reward = 0
        self.surface = pygame.Surface((self.width, self.height), flags=pygame.SRCALPHA)
        self.acceleration = 0.1

        self.target_i = 0
        pygame.draw.rect(self.surface, pygame.Color("red"), [0, 0, self.width, self.height])
    def rotation(self,angle):
        self.direction += self.rotationspeed*angle
        self.direction %= 360
    def speedadjust(self,action):
        self.speed += self.acceleration*action
        if(self.speed > self.max_speed):
            self.speed = self.max_speed
        elif(self.speed < -self.max_speed):
            self.speed = -self.max_speed
class Game:
    def __init__(self, N_agent):
        self.Width = map_width*block_size
        self.Height = map_height*block_size
        self.screen = pygame.display.set_mode((self.Width, self.Height))
        pygame.display.set_caption('Game')
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.N_agent = N_agent
        self.__reset__()
        self.maxFrames= 1000
        self.targets = []
        self.target_radius = 15
        self.mutation_rate = 0.1
        self.crossover_rate = 0.1
        self.mutation_strength = 0.1

    def __reset__(self):
        self.cars = [Car() for _ in range(self.N_agent)]
        self.frames = 0
        self.done = False

    def __gamestep__(self,actions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.done = True
                    return
                if event.key == pygame.K_RIGHT:
                    self.mutation_rate +=0.02
                if event.key == pygame.K_LEFT:
                    self.mutation_rate -=0.02
                if event.key == pygame.K_UP:
                    self.crossover_rate +=0.02
                if event.key == pygame.K_DOWN:
                    self.crossover_rate -=0.02
                if event.key == pygame.K_x:
                    self.mutation_strength += 0.02
                if event.key == pygame.K_z:
                    self.mutation_strength -= 0.02
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.targets.append(pygame.mouse.get_pos())
        #rest of the logic
        self.done = True
        self.frames += 1
        for i in range(0,len(self.cars)):
            if(self.cars[i].alive):
                self.done = False
                self.cars[i].speedadjust(actions[i][0])
                self.cars[i].rotation(actions[i][1])
                self.move(self.cars[i])
                self.collisiondetection(self.cars[i])
                self.cars[i].reward+=(1 if self.cars[i].speed>self.cars[i].max_speed/2 else -1)
        if(self.frames >= self.maxFrames):
            self.done = True
        self.draw()
        self.clock.tick(self.fps)
    def move(self, car):
        car.horizontal_velocity = car.speed * math.cos(math.radians(car.direction))
        car.vertical_velocity = car.speed * -math.sin(math.radians(car.direction))
        car.x += car.horizontal_velocity
        car.y += car.vertical_velocity
        car.x = int(car.x)
        car.y = int(car.y)
    def draw(self):
        self.screen.blit(map_surface,(0,0))
        for car in self.cars:
            rotated_surface = pygame.transform.rotate(car.surface, car.direction)
            rect = rotated_surface.get_rect(center=(car.x, car.y))
            self.screen.blit(rotated_surface, rect.topleft)
            pygame.draw.circle(self.screen, (0, 255, 0), [car.x, car.y], 4)
        for target in self.targets:
            pygame.draw.circle(self.screen, (0, 255, 0), target, 15)
        text = font.render("mutation_rate:"+str(self.mutation_rate), True, (255,0,0))
        self.screen.blit(text, (0,0))
        text2 = font.render("crossover_rate:" + str(self.crossover_rate), True, (255,0,0))
        self.screen.blit(text2, (0,40))
        text3 = font.render("mutation_strength"+str(self.mutation_strength), True, (255,0,0))
        self.screen.blit(text3, (0,80))
        pygame.display.flip()
    def collisiondetection(self,car):
        x, y = car.x // block_size, car.y // block_size
        car_vertices = get_rect_vertices(car.x, car.y, car.width, car.height, car.direction)
        car_polygon = PolygonCollision.shape.Shape(vertices=car_vertices)
        Range = 5
        if(car.target_i<len(self.targets)):
            target_circle = PolygonCollision.shape.Shape([self.targets[car.target_i]],radius=self.target_radius)
            if(target_circle.collide(car_polygon)):
                car.reward+=10
                car.target_i+=1
        for i in range(max(x - Range, 0), min(x + Range, binary_matrix.shape[1] - 1)):
            for j in range(max(y - Range, 0), min(y + Range, binary_matrix.shape[0] - 1)):
                if binary_matrix[j][i] == 1:
                    block_vertices = [
                        (i * block_size, j * block_size),
                        ((i + 1) * block_size, j * block_size),
                        ((i + 1) * block_size, (j + 1) * block_size),
                        (i * block_size, (j + 1) * block_size)
                    ]
                    block_polygon = PolygonCollision.shape.Shape(vertices=block_vertices)
                    if car_polygon.collide(block_polygon):
                        car.alive = False
                        return
        pass
    def raycast(self,car,max_length = 200,step_size = 2):
        angles = [0, -45, 45, -90, 90]
        result = []
        car_center_x = car.x + car.width // 2
        car_center_y = car.y + car.height // 2

        for angle in angles:
            ray_angle = car.direction + angle
            ray_angle_rad = math.radians(ray_angle)
            length = 0
            hit = False
            while length < max_length and not hit:
                test_x = int(car_center_x + length * math.cos(ray_angle_rad))
                test_y = int(car_center_y - length * math.sin(ray_angle_rad))
                if (0 <= test_x < binary_matrix.shape[1] * block_size) and (
                        0 <= test_y < binary_matrix.shape[0] * block_size):
                    matrix_x = test_x // block_size
                    matrix_y = test_y // block_size
                    if binary_matrix[matrix_y][matrix_x] == 1:
                        hit = True
                else:
                    hit = True
                length += step_size
            result.append(length/max_length)
        return np.array(result)

