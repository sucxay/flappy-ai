import pygame 
import random 
import neat 
import os 

pygame.init()
WIDTH  = 800
HEIGHT = 600

WIN = pygame.display.set_mode((WIDTH , HEIGHT))
pygame.display.set_caption('Flappy bird AI')

BIRD_X = 150
BIRD_SIZE = 25 
FONT = pygame.font.SysFont("arial" ,30)

PIPE_WIDTH = 70
PIPE_GAP = 170
PIPE_SPEED = 5

GRAVITY = 0.5
JUMP = -9


class bird:
    def __init__ (self, y):
        self.x = BIRD_X
        self.y = y 
        self.velocity = 0
        self.alive = True
        self.score = 0

    def jump(self):
        self.velocity = JUMP
    def move(self):
        self.velocity+=GRAVITY
        self.y +=self.velocity
    def draw(self):
        pygame.draw.circle(
            WIN , (225 ,220,0),
            (int(self.x),int(self.y)),
            BIRD_SIZE
        )
    def get_rect(self):
        return pygame.Rect(
            self.x - BIRD_SIZE,
            self.y - BIRD_SIZE,
            BIRD_SIZE * 2,
            BIRD_SIZE * 2
        )
class Pipe:

    def __init__(self, x):
        self.x = x

        self.gap_y = random.randint(
            150,
            HEIGHT - 150
        )

        self.passed = False

    def move(self):
        self.x -= PIPE_SPEED

    def draw(self):

        top_height = self.gap_y - PIPE_GAP // 2
        bottom_y = self.gap_y + PIPE_GAP // 2

        pygame.draw.rect(
            WIN,
            (0, 200, 0),
            (self.x, 0, PIPE_WIDTH, top_height)
        )
        pygame.draw.rect(
            WIN,
            (0, 200, 0),
            (
                self.x,
                bottom_y,
                PIPE_WIDTH,
                HEIGHT - bottom_y
            )
        )

    def collides(self, bird):

        bird_rect = bird.get_rect()

        top_pipe = pygame.Rect(
            self.x,
            0,
            PIPE_WIDTH,
            self.gap_y - PIPE_GAP // 2
        )

        bottom_pipe = pygame.Rect(
            self.x,
            self.gap_y + PIPE_GAP // 2,
            PIPE_WIDTH,
            HEIGHT
        )

        return (
            bird_rect.colliderect(top_pipe)
            or bird_rect.colliderect(bottom_pipe)
        )

def draw_window(birds, pipes, generation):

    WIN.fill((135, 206, 235))

    for pipe in pipes:
        pipe.draw()

    for bird in birds:
        if bird.alive:
            bird.draw()

    alive = sum(
        1 for bird in birds
        if bird.alive
    )

    text = FONT.render(
        f"Generation: {generation}   Alive: {alive}",
        True,
        (0, 0, 0)
    )

    WIN.blit(text, (20, 20))

    pygame.display.update()



def eval_genomes(genomes, config):

    birds = []
    networks = []
    genome_list = []

    for _, genome in genomes:

        genome.fitness = 0

        network = neat.nn.FeedForwardNetwork.create(
            genome,
            config
        )

        networks.append(network)
        birds.append(bird(HEIGHT // 2))
        genome_list.append(genome)

    pipes = [Pipe(WIDTH)]

    clock = pygame.time.Clock()

    running = True

    while running and len(birds) > 0:

        clock.tick(60)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Find the next pipe
        pipe = None

        for p in pipes:

            if p.x + PIPE_WIDTH > BIRD_X:
                pipe = p
                break

        if pipe is None:
            pipe = pipes[0]

        # ------------------------------------------
        # AI decisions
        # ------------------------------------------

        for i, bird in enumerate(birds):

            if not bird.alive:
                continue

            genome = genome_list[i]
            net = networks[i]

            gap_top = pipe.gap_y - PIPE_GAP / 2
            gap_bottom = pipe.gap_y + PIPE_GAP / 2

            inputs = (
                bird.y / HEIGHT,
                bird.velocity / 10,
                (pipe.x - bird.x) / WIDTH,
                gap_top / HEIGHT,
                gap_bottom / HEIGHT
            )

            output = net.activate(inputs)

            if output[0] > 0.5:
                bird.jump()

            bird.move()

            # Survival reward
            genome.fitness += 0.1

        # ------------------------------------------
        # Move pipes
        # ------------------------------------------

        for p in pipes:
            p.move()

        # Add new pipe
        if pipes[-1].x < WIDTH - 350:
            pipes.append(Pipe(WIDTH))

        # Remove old pipes
        if pipes[0].x < -PIPE_WIDTH:
            pipes.pop(0)

        # ------------------------------------------
        # Collision / scoring
        # ------------------------------------------

        for i, bird in enumerate(birds):

            if not bird.alive:
                continue

            genome = genome_list[i]

            # Ground / ceiling
            if (
                bird.y - BIRD_SIZE <= 0
                or bird.y + BIRD_SIZE >= HEIGHT
            ):
                bird.alive = False
                genome.fitness -= 5
                continue

            # Pipe collision
            for p in pipes:

                if p.collides(bird):

                    bird.alive = False
                    genome.fitness -= 5
                    break

            if not bird.alive:
                continue

            # Passed pipe
            for p in pipes:

                if (
                    not p.passed
                    and p.x + PIPE_WIDTH < bird.x
                ):

                    p.passed = True

                    bird.score += 1
                    genome.fitness += 10

        # Remove dead birds
        if not any(bird.alive for bird in birds):
            running = False

        draw_window(
            birds,
            pipes,
            eval_genomes.generation
        )

    eval_genomes.generation += 1


eval_genomes.generation = 1


# --------------------------------------------------
# NEAT configuration
# --------------------------------------------------

def run_neat():

    config_path = os.path.join(
        os.path.dirname(__file__),
        "config-feedforward.txt"
    )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)

    population.add_reporter(
        neat.StdOutReporter(True)
    )

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(
        eval_genomes,
        100
    )

    print("\nBest genome:")
    print(winner)


if __name__ == "__main__":
    run_neat()




    