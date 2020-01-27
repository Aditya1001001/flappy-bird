import pygame
import neat
import time
import os
import random

WIN_WIDTH = 500
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))), pygame.transform.scale2x(
    pygame.image.load(os.path.join("images", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird3.png")))]
BASE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("images", "base.png")))
PIPE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("images", "pipe.png")))
BG_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("images", "bg.png")))

pygame.font.init()
GAME_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tick_count = 0
        self.tilt = 0
        self.vel = 0
        self.height = y
        self.img_num = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel*self.tick_count + 1.5*self.tick_count**2

        # terminal velocity
        if d >= 16:
            d = 16

        # how high the jumps are
        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt = - self.ROT_VEL

    def draw(self, win):
        self.img_num += 1
        # choosing which bird image to show
        # making the bird flap its wing
        if self.img_num < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_num < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_num < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_num < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_num < self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_num = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_num = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self. y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.height = 0
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        else:
            return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        # once the first image is out of the frame we cycle it to the back
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        # when the second image is out of the frame we cycle it to the back
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        ''' draws the two base images'''
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, bird, pipes, base, score):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    text = GAME_FONT.render("Score-   " + str(score), 1, (255, 255, 255))
    # keep changing the location of the score based on the number of digits
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    base.draw(win)
    bird.draw(win)
    pygame.display.update()


def main(genomes, config):
    nets = []
    birds = []
    ge = []

    for g in genomes:
        net = neat.nn.FeedForwardNetwork(g, config)
        nets.append(net)
        birds.append(Bird(240, 360))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    game_running = True
    score = 0
    while game_running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
        # bird.move()
        add_pipe = False
        removed = []
        for pipe in pipes:
            for x,bird in enumerate(birds):
                if pipe.collide(bird):
                    # decreasing the fitness of birds that hit a pipe 
                    ge[x].fitness -=1
                    birds.remove(bird)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removed.append(pipe)
            pipe.move()
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))

        for r in removed:
            pipes.remove(r)
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730:
                birds.remove(bird)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(win, bird, pipes, base, score)
    pygame.quit()
    quit()


main()


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpecieSet, neat.DefaultStagnation, config_path)
    
    pop = neat.Population(config)


    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticReporter()
    pop.add_reporter(stats)
    
    winner = pop.run(main,50)


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "config.txt")
