import pygame
from env import Snake, Apple, WIDTH, TILE_WIDTH, RES
from agent import Agent
import numpy as np
pygame.init()

screen = pygame.display.set_mode(RES)
pygame.display.set_caption("training...")

snake = Snake()
apple = Apple(snake)

agent = Agent(WIDTH)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    game_frame = np.zeros(shape = (WIDTH, WIDTH, 1))
    for tile in snake.body[1:]:
        game_frame[tile[1], tile[0], 0] = 1 # body tile
    game_frame[snake.body[0][1], snake.body[0][0], 0] = 2 # Snake's head
    game_frame[apple.position[1], apple.position[0], 0] = 3 # Apple

    new_direction = agent.predict_action(game_frame)
    snake.move(new_direction)
    reward = 0
    if snake.check_eat(apple): reward = 1
    if snake.check_death():
        reward = -1
        quit() # temporary

    apple.draw(screen)
    snake.draw(screen)

    pygame.display.flip()

    pygame.time.wait(100)
