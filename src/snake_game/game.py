import pygame
import random
import numpy as np
from pygame.math import Vector2

dir_path = 'assets/game_images'


class SNAKE:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10), Vector2(2, 10), Vector2(1, 10)]
        self.direction = Vector2(1, 0)
        self.new_block = False
        self.head = None
        self.tail = None
        self.head_up = None
        self.head_down = None
        self.head_right = None
        self.head_left = None
        self.tail_up = None
        self.tail_down = None
        self.tail_right = None
        self.tail_left = None
        self.body_vertical = None
        self.body_horizontal = None
        self.body_tr = None
        self.body_tl = None
        self.body_br = None
        self.body_bl = None

    def load_images(self):
        self.head_up = pygame.transform.scale(pygame.image.load(f'{dir_path}/head_up.png').convert_alpha(),
                                              (self.cell_size, self.cell_size))
        self.head_down = pygame.transform.scale(pygame.image.load(f'{dir_path}/head_down.png').convert_alpha(),
                                                (self.cell_size, self.cell_size))
        self.head_right = pygame.transform.scale(pygame.image.load(f'{dir_path}/head_right.png').convert_alpha(),
                                                 (self.cell_size, self.cell_size))
        self.head_left = pygame.transform.scale(pygame.image.load(f'{dir_path}/head_left.png').convert_alpha(),
                                                (self.cell_size, self.cell_size))
        self.tail_up = pygame.transform.scale(pygame.image.load(f'{dir_path}/tail_up.png').convert_alpha(),
                                              (self.cell_size, self.cell_size))
        self.tail_down = pygame.transform.scale(pygame.image.load(f'{dir_path}/tail_down.png').convert_alpha(),
                                                (self.cell_size, self.cell_size))
        self.tail_right = pygame.transform.scale(pygame.image.load(f'{dir_path}/tail_right.png').convert_alpha(),
                                                 (self.cell_size, self.cell_size))
        self.tail_left = pygame.transform.scale(pygame.image.load(f'{dir_path}/tail_left.png').convert_alpha(),
                                                (self.cell_size, self.cell_size))
        self.body_vertical = pygame.transform.scale(pygame.image.load(f'{dir_path}/body_vertical.png').convert_alpha(),
                                                    (self.cell_size, self.cell_size))
        self.body_horizontal = pygame.transform.scale(
            pygame.image.load(f'{dir_path}/body_horizontal.png').convert_alpha(), (self.cell_size, self.cell_size))
        self.body_tr = pygame.transform.scale(pygame.image.load(f'{dir_path}/body_tr.png').convert_alpha(),
                                              (self.cell_size, self.cell_size))
        self.body_tl = pygame.transform.scale(pygame.image.load(f'{dir_path}/body_tl.png').convert_alpha(),
                                              (self.cell_size, self.cell_size))
        self.body_br = pygame.transform.scale(pygame.image.load(f'{dir_path}/body_br.png').convert_alpha(),
                                              (self.cell_size, self.cell_size))
        self.body_bl = pygame.transform.scale(pygame.image.load(f'{dir_path}/body_bl.png').convert_alpha(),
                                              (self.cell_size, self.cell_size))

    def draw_snake(self, screen):
        self.update_head_graphics()
        self.update_tail_graphics()
        for index, block in enumerate(self.body):
            x_pos = int(block.x * self.cell_size)
            y_pos = int(block.y * self.cell_size)
            block_rect = pygame.Rect(x_pos, y_pos, self.cell_size, self.cell_size)
            if index == 0:
                screen.blit(self.head, block_rect)
            elif index == len(self.body) - 1:
                screen.blit(self.tail, block_rect)
            else:
                previous_block = self.body[index + 1] - block
                next_block = self.body[index - 1] - block
                if previous_block.x == next_block.x:
                    screen.blit(self.body_vertical, block_rect)
                elif previous_block.y == next_block.y:
                    screen.blit(self.body_horizontal, block_rect)
                else:
                    if (previous_block.x == -1 and next_block.y == -1) or (
                            previous_block.y == -1 and next_block.x == -1):
                        screen.blit(self.body_tl, block_rect)
                    elif (previous_block.x == -1 and next_block.y == 1) or (
                            previous_block.y == 1 and next_block.x == -1):
                        screen.blit(self.body_bl, block_rect)
                    elif (previous_block.x == 1 and next_block.y == -1) or (
                            previous_block.y == -1 and next_block.x == 1):
                        screen.blit(self.body_tr, block_rect)
                    elif (previous_block.x == 1 and next_block.y == 1) or (previous_block.y == 1 and next_block.x == 1):
                        screen.blit(self.body_br, block_rect)

    def update_head_graphics(self):
        if len(self.body) > 1:
            head_relation = self.body[1] - self.body[0]
            if head_relation == Vector2(1, 0):
                self.head = self.head_left
            elif head_relation == Vector2(-1, 0):
                self.head = self.head_right
            elif head_relation == Vector2(0, 1):
                self.head = self.head_up
            elif head_relation == Vector2(0, -1):
                self.head = self.head_down
        else:
            self.head = self.head_right

    def update_tail_graphics(self):
        if len(self.body) > 1:
            tail_relation = self.body[-2] - self.body[-1]
            if tail_relation == Vector2(1, 0):
                self.tail = self.tail_left
            elif tail_relation == Vector2(-1, 0):
                self.tail = self.tail_right
            elif tail_relation == Vector2(0, 1):
                self.tail = self.tail_up
            elif tail_relation == Vector2(0, -1):
                self.tail = self.tail_down
        else:
            self.tail = self.tail_right

    def move_snake(self):
        if self.direction == Vector2(0, 0):
            return
        if self.new_block:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

    def reset(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10), Vector2(2, 10), Vector2(1, 10)]
        self.direction = Vector2(1, 0)


class FRUIT:
    def __init__(self, cell_size, cell_number):
        self.x = None
        self.pos = None
        self.y = None
        self.cell_size = cell_size
        self.cell_number = cell_number
        self.apple = None
        self.randomize()

    def load_images(self):
        self.apple = pygame.transform.scale(pygame.image.load(f'{dir_path}/apple.png').convert_alpha(),
                                            (self.cell_size, self.cell_size))

    def draw_fruit(self, screen):
        fruit_rect = pygame.Rect(int(self.pos.x * self.cell_size), int(self.pos.y * self.cell_size), self.cell_size,
                                 self.cell_size)
        screen.blit(self.apple, fruit_rect)

    def randomize(self):
        self.x = random.randint(0, self.cell_number - 1)
        self.y = random.randint(0, self.cell_number - 1)
        self.pos = Vector2(self.x, self.y)


class SnakeEnv:
    def __init__(self, cell_number=25, cell_size=30, rendering=False):
        self.cell_number = cell_number
        self.cell_size = cell_size
        self.rendering = rendering
        self.state_grid_size = cell_number + 2
        if self.rendering:
            pygame.init()
            self.screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
            self.clock = pygame.time.Clock()
            self.game_font = pygame.font.SysFont("arial", 25)
        self.snake = SNAKE(cell_size)
        self.fruit = FRUIT(cell_size, cell_number)
        if self.rendering:
            self.snake.load_images()
            self.fruit.load_images()
            self.apple = pygame.transform.scale(pygame.image.load(f'{dir_path}/apple.png').convert_alpha(),
                                                (self.cell_size, self.cell_size))
        self.action_space = [0, 1, 2]
        self.new_direction_table = {0: [0, 2, 3], 1: [1, 3, 2], 2: [2, 1, 0], 3: [3, 0, 1]}
        self.steps_since_last_fruit = 0
        self.max_steps_without_fruit = self.cell_size * self.cell_size * len(self.snake.body)

    def reset(self):
        self.snake.reset()
        self.fruit.randomize()
        self.snake.direction = Vector2(1, 0)
        self.steps_since_last_fruit = 0
        return self._get_state()

    def step(self, action):
        self.steps_since_last_fruit += 1
        previous_state = self._get_state()
        current_direction_vector = self.snake.direction
        current_index = self.vector_to_index(current_direction_vector)
        new_index = self.new_direction_table[current_index][action]
        self.snake.direction = self.index_to_vector(new_index)
        self.snake.move_snake()
        reward = 0
        done = False
        if (not 0 <= self.snake.body[0].x < self.cell_number or
                not 0 <= self.snake.body[0].y < self.cell_number or
                self.snake.body[0] in self.snake.body[1:]):
            reward = -1
            done = True
            state = previous_state
            score = len(self.snake.body) - 5
            print(f"Snake died! Score: {score}")
            if self.rendering:
                self.render()
            return state, reward, done, {}
        if self.snake.body[0] == self.fruit.pos:
            self.fruit.randomize()
            self.snake.add_block()
            reward = 0.2
            self.steps_since_last_fruit = 0
            while self.fruit.pos in self.snake.body:
                self.fruit.randomize()
        if self.steps_since_last_fruit >= self.max_steps_without_fruit:
            reward = -0.2
            done = True
            score = len(self.snake.body) - 5
            print(f"Snake died! Score: {score}")
        state = self._get_state()
        if self.rendering:
            self.render()
        return state, reward, done, {}

    def compute_distance_to_obstacle(self, head, direction, max_distance=5):
        pos = head
        for i in range(1, max_distance + 1):
            pos = pos + direction
            if not (0 <= pos.x < self.cell_number and 0 <= pos.y < self.cell_number):
                return i
            if pos in self.snake.body:
                return i
        return max_distance + 1

    def _get_state(self) -> np.ndarray:
        state = np.zeros(11, dtype=np.float32)
        head = self.snake.body[0]
        direction = self.snake.direction
        current_dir_idx = self.vector_to_index(direction)
        straight_dir_idx = self.new_direction_table[current_dir_idx][0]
        right_dir_idx = self.new_direction_table[current_dir_idx][2]
        left_dir_idx = self.new_direction_table[current_dir_idx][1]
        straight_dir = self.index_to_vector(straight_dir_idx)
        right_dir = self.index_to_vector(right_dir_idx)
        left_dir = self.index_to_vector(left_dir_idx)
        d_straight = self.compute_distance_to_obstacle(head, straight_dir)
        danger_straight = 3 if d_straight == 1 else 2 if d_straight == 2 else 1 if 3 <= d_straight <= 5 else 0
        state[0] = danger_straight / 3.0
        d_right = self.compute_distance_to_obstacle(head, right_dir)
        danger_right = 3 if d_right == 1 else 2 if d_right == 2 else 1 if 3 <= d_right <= 5 else 0
        state[1] = danger_right / 3.0
        d_left = self.compute_distance_to_obstacle(head, left_dir)
        danger_left = 3 if d_left == 1 else 2 if d_left == 2 else 1 if 3 <= d_left <= 5 else 0
        state[2] = danger_left / 3.0
        state[3] = 1 if direction == Vector2(-1, 0) else 0  # Left
        state[4] = 1 if direction == Vector2(1, 0) else 0  # Right
        state[5] = 1 if direction == Vector2(0, -1) else 0  # Up
        state[6] = 1 if direction == Vector2(0, 1) else 0  # Down
        food_pos = self.fruit.pos
        state[7] = (head.x - food_pos.x) / self.cell_number if food_pos.x < head.x else 0  # Food left
        state[8] = (food_pos.x - head.x) / self.cell_number if food_pos.x > head.x else 0  # Food right
        state[9] = (head.y - food_pos.y) / self.cell_number if food_pos.y < head.y else 0  # Food up
        state[10] = (food_pos.y - head.y) / self.cell_number if food_pos.y > head.y else 0  # Food down
        return state

    def render(self):
        self.screen.fill((175, 215, 70))
        self.draw_grass()
        self.fruit.draw_fruit(self.screen)
        self.snake.draw_snake(self.screen)
        self.draw_score()
        pygame.display.update()
        self.clock.tick(30)

    def draw_grass(self):
        grass_color = (167, 209, 61)
        for row in range(self.cell_number):
            if row % 2 == 0:
                for col in range(self.cell_number):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size,
                                                 self.cell_size)
                        pygame.draw.rect(self.screen, grass_color, grass_rect)
            else:
                for col in range(self.cell_number):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size,
                                                 self.cell_size)
                        pygame.draw.rect(self.screen, grass_color, grass_rect)

    def draw_score(self):
        score_text = str(len(self.snake.body) - 5)
        score_surface = self.game_font.render(score_text, True, (56, 74, 12))
        score_x = int(self.cell_size * self.cell_number - 60)
        score_y = int(self.cell_size * self.cell_number - 40)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        apple_rect = self.apple.get_rect(midright=(score_rect.left, score_rect.centery))
        bg_rect = pygame.Rect(apple_rect.left, apple_rect.top, apple_rect.width + score_rect.width + 6,
                              apple_rect.height)
        pygame.draw.rect(self.screen, (167, 209, 61), bg_rect)
        self.screen.blit(score_surface, score_rect)
        self.screen.blit(self.apple, apple_rect)
        pygame.draw.rect(self.screen, (56, 74, 12), bg_rect, 2)

    @staticmethod
    def vector_to_index(vector):
        if vector == Vector2(0, -1):
            return 0  # up
        elif vector == Vector2(0, 1):
            return 1  # down
        elif vector == Vector2(-1, 0):
            return 2  # left
        elif vector == Vector2(1, 0):
            return 3  # right
        return 3  # default to right

    @staticmethod
    def index_to_vector(index):
        vectors = [Vector2(0, -1), Vector2(0, 1), Vector2(-1, 0), Vector2(1, 0)]
        return vectors[index]

    def close(self):
        if self.rendering:
            pygame.quit()
