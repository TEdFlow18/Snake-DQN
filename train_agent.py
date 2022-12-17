import pygame
from env import Snake, Apple, WIDTH, TILE_WIDTH, RES
from agent import Agent
import numpy as np
pygame.init()

screen = pygame.display.set_mode(RES)
pygame.display.set_caption("training...")

snake = Snake()
apples = []
for _ in range(5):
    apples.append(Apple(snake))

agent = Agent(WIDTH)

def get_state(snake, apples):
    game_frame = np.zeros(shape = (WIDTH, WIDTH, 1))
    for tile in snake.body[1:]:
        game_frame[tile[1], tile[0], 0] = 1 # body tile
    game_frame[snake.body[0][1], snake.body[0][0], 0] = 2 # Snake's head
    for apple in apples:
        game_frame[apple.position[1], apple.position[0], 0] = 3 # Apples
    return game_frame

running = True

nb_frame = 0
nb_turn = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))


    game_frame = get_state(snake, apples)
    new_direction, action = agent.predict_action(game_frame)
    snake.move(new_direction)

    reward = 0
    done = False
    for apple in apples:
        if snake.check_eat(apple): reward = 1

    if snake.check_death():
        reward = -1
        done = True
        snake = Snake()
        apples = []
        for _ in range(5):
            apples.append(Apple(snake))
        nb_turn = 0

    next_state = get_state(snake, apples)

    for apple in apples:
        apple.draw(screen)
    snake.draw(screen)

    agent.add_observation(game_frame, action, reward, next_state, done)

    if nb_frame > 200:
        agent.train()

    pygame.display.flip()

    if nb_turn > 50:
        snake = Snake()
        apples = []
        for _ in range(5):
            apples.append(Apple(snake))
        nb_turn = 0

    nb_frame += 1
    if nb_frame > 10000:
        agent.save()
        nb_frame = 51

    nb_turn += 1

    # pygame.time.wait(100)
