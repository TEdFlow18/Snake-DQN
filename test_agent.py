import pygame
from env import Snake, Apple, WIDTH, TILE_WIDTH, RES
from agent import Agent
import numpy as np
import sys
pygame.init()

if len(sys.argv) != 1:
    raise Exception("One argument must be specified : the path to tha AI's model.")

screen = pygame.display.set_mode(RES)
pygame.display.set_caption("training...")

snake = Snake()
apple = Apple(snake)

agent = Agent(WIDTH)

agent.load_model(sys.argv[1])

def get_state(snake, apple):
    game_frame = np.zeros(shape = (WIDTH, WIDTH, 1))
    for tile in snake.body[1:]:
        game_frame[tile[1], tile[0], 0] = 1 # body tile
    game_frame[snake.body[0][1], snake.body[0][0], 0] = 2 # Snake's head
    game_frame[apple.position[1], apple.position[0], 0] = 3 # Apple
    return game_frame

running = True

nb_frame = 0
nb_turn = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))


    game_frame = get_state(snake, apple)
    new_direction, action = agent.predict_action(game_frame)
    snake.move(new_direction)

    if snake.check_eat(apple): reward = 1

    if snake.check_death():
        snake = Snake()
        apple = Apple(snake)
        nb_turn = 0

    next_state = get_state(snake, apple)

    apple.draw(screen)
    snake.draw(screen)

    pygame.display.flip()

    # pygame.time.wait(100)
