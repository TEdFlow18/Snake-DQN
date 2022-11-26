import pygame
import random

WIDTH = 20
TILE_WIDTH = 20
RES = (WIDTH*TILE_WIDTH, WIDTH*TILE_WIDTH)


class Snake:
    def __init__(self):
        self.direction = (1, 0)
        self.body = [(WIDTH//2, WIDTH//2)]
        for i in range(1, 4):
            self.body.append((self.body[0][0]-i, self.body[0][1]))

    def move(self, new_direction = None):
        if new_direction is not None:
            self.direction = new_direction

        self.body.pop()
        self.body.insert(0,
            (self.body[0][0]+self.direction[0],
            self.body[0][1]+self.direction[1]))

    def check_eat(self, apple): # Check if the snake has eaten the apple
        if self.body[0][0] == apple.position[0] and self.body[0][1] == apple.position[1]:
            apple.new_position(self)
            self.body.append((self.body[-1][0], self.body[-1][1])) # Make the snake longer
            return True
        return False

    def check_death(self): # Check if the snake should die
        head = self.body[0]
        if head[0] > WIDTH-1 or head[0] < 0 or head[1] > WIDTH-1 or head[1] < 0:
            return True
        for tile in self.body[1:]:
            if tile[0] == head[0] and tile[1] == head[1]:
                return True

        return False

    def draw(self, screen):
        for tile in self.body:
            pygame.draw.rect(
                screen,
                (0, 255, 0),
                (tile[0]*TILE_WIDTH,
                tile[1]*TILE_WIDTH,
                TILE_WIDTH, TILE_WIDTH),
            )


class Apple:
    def __init__(self, snake):
        self.new_position(snake)

    def new_position(self, snake):
        while True:
            self.position = (random.randint(0, WIDTH-1), random.randint(0, WIDTH-1))
            for tile in snake.body:
                if tile[0] == self.position[0] and tile[1] == self.position[1]:
                    break
            else:
                return self.position

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            (255, 0, 0),
            (self.position[0]*TILE_WIDTH+TILE_WIDTH/2,
            self.position[1]*TILE_WIDTH+TILE_WIDTH/2),
            TILE_WIDTH/2
        )


def main():
    pygame.init()

    screen = pygame.display.set_mode(RES)
    pygame.display.set_caption("Snake AI")

    snake = Snake()
    apple = Apple(snake)

    running = True
    while running:
        direction = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    direction = (-1, 0)
                elif event.key == pygame.K_RIGHT:
                    direction = (1, 0)
                elif event.key == pygame.K_UP:
                    direction = (0, -1)
                elif event.key == pygame.K_DOWN:
                    direction = (0, 1)


        screen.fill((0, 0, 0))

        snake.move(direction)
        if snake.check_eat(apple):
            print("+1")
        if snake.check_death():
            print("-1")
            quit()

        apple.draw(screen)
        snake.draw(screen)

        pygame.display.flip()

        pygame.time.wait(100)

        # quit()

if __name__ == "__main__":
    main()