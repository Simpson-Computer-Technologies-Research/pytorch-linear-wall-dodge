##
## Neural network that controls a dot (reinformcement learning)
##
## This will use pygame to draw the environment and the dot
##
## The dot will try to dodge the walls
##
## The dot will have 8 sensors that will detect the distance to the wall
## The dot will have 4 outputs that will be the movement of the dot
##
import torch, pygame, random, math
import torch.nn as nn
from typing import List

# Constants
WIDTH = 800
HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dot")
clock = pygame.time.Clock()


# Neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_channels, input_channels * 2)
        self.fc2 = nn.Linear(input_channels * 2, input_channels * 2)
        self.fc3 = nn.Linear(input_channels * 2, output_channels)

        self.mutation_rate = 0.01

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def mutate(self):
        for param in self.parameters():
            param.data += self.mutation_rate * torch.randn(param.size())


# Wall class
class Wall:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.width = 10
        self.height = 10

    def draw(self):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height))


# Dot class
class Dot:
    def __init__(self, num_sensors: int):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 2
        self.brain = NeuralNetwork(input_channels=num_sensors, output_channels=4)
        self.sensors = [0 for _ in range(num_sensors)]
        self.dead = False
        self.deaths = 0

    # update the sensors
    def update_sensors(self, walls: List[Wall]):
        for i in range(len(self.sensors)):
            self.sensors[i] = math.sqrt(
                (self.x - walls[i].x) ** 2 + (self.y - walls[i].y) ** 2
            )

    # move the dot
    def move(self, walls: List[Wall]):
        self.update_sensors(walls)

        output = self.brain(torch.tensor(self.sensors).float())

        self.x += int(output[0].item() * self.speed)
        self.y += int(output[1].item() * self.speed)

        # if we hit a wall tell the brain it's dead
        for wall in walls:
            if (
                wall.x <= self.x <= wall.x + wall.width
                and wall.y <= self.y <= wall.y + wall.height
            ):
                self.dead = True

        # mutate the brain parameters if it's dead
        if self.dead:
            self.brain.mutate()

            self.x = WIDTH // 2
            self.y = HEIGHT // 2
            self.dead = False
            self.deaths += 1

        return output

    def draw(self):
        pygame.draw.circle(screen, RED, (self.x, self.y), 5)


# Main loop
num_walls = 100
walls = [Wall() for _ in range(num_walls)]
dot = Dot(num_sensors=num_walls)
running = True

while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for wall in walls:
        wall.draw()

    output = dot.move(walls)
    dot.draw()

    print(f"Attempt #{dot.deaths + 1}: {output}")

    # if the dot reached the edge, then it won
    if dot.x <= 0 or dot.x >= WIDTH or dot.y <= 0 or dot.y >= HEIGHT:
        print(f"The dot successfully dodged the walls in {dot.deaths + 1} attemps!")
        running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
