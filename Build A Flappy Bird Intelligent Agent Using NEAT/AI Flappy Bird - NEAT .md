
# How to Built A Flappy Bird AI Using NEAT

In this project, I will use Pygame to build Flappy Bird and apply NEAT algorithm to train an intelligence agent. NEAT, also known as NeuroEvolution of Augmenting Topologies, is a genetic algorithm designed to evolve artificial neural network topologies in an efficient way. It’s an awesome technique developed by Kenneth Stanley in 2002 that addressed some challenges of Topology and Weight Evolving Artiﬁcial Neural Networks (TWEANN). 

Danny Zhu

May 20, 2020

## Overview

**Game Initialization:**
- Pygame Initialization
- Game Parameters Setting

**Game Objects:**
- Build A Bird
- Build A Pipe
- Build A Floor

**Game Functions:**
- Check Collision
- Draw Window

**NEAT Initialization:**
- NEAT Parameters Setting
- Configuration File

**NEAT Functions:**
- Get Input Index
- Visualization Functions

**Finalization**
- Main Game Loop
- Run NEAT

### Pygame Initialization


```python
#import packages to build the game
from __future__ import print_function
import pygame
import time
import os
import random

#initialize pygame
pygame.init()
```

    pygame 1.9.6
    Hello from the pygame community. https://www.pygame.org/contribute.html
    




    (6, 0)



### Game Parameters Setting


```python
#set up the screen to display the game
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 550
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

#set up the font
FONT = pygame.font.SysFont('comicsansms', 20)
FONT_COLOR = (255, 255, 255) #white font

#load the required images
BIRD_IMGS = [pygame.image.load('Flappy Bird.png'),
             pygame.image.load('Flappy Bird Wings Up.png'),
             pygame.image.load('Flappy Bird Wings Down.png')]
BOTTOM_PIPE_IMG = pygame.image.load('Super Mario pipe.png')
TOP_PIPE_IMG = pygame.transform.flip(BOTTOM_PIPE_IMG, False, True) #flip the image of the bottom pipe to get the image for the pipe on the top
FLOOR_IMG = pygame.image.load('Stone Floor.png')
BG_IMG = pygame.transform.scale(pygame.image.load('City Skyline.png'), (SCREEN_WIDTH, SCREEN_HEIGHT))

#set the game options
FPS = 30 #run the game at rate FPS, the speed at which images are shown
max_score = 100 #the maximum score of the game before we break the loop
```


```python
#floor options
floor_velocity = 5 #the horizontal moving velocity of the floor, this should equal to pipe_velocity
floor_starting_y_position = 500 #the starting y position of the floor

#pipe options
pipe_max_num = 100 #the maximum number of pipes in this game
pipe_vertical_gap = 150 #the gap between the top pipe and the bottom pipe, the smaller the number, the harder the game
pipe_horizontal_gap = 200 #the gap between two sets of pipes
pipe_velocity = 5 #the horizontal moving velocity of the pipes, this should equal to floor_velocity
top_pipe_min_height = 100 #the minimum height of the top pipe (carefully set this number)
top_pipe_max_height = 300 #the maximum height of the top pipe (carefully set this number)
pipe_starting_x_position = 500 #the starting x position of the first pipe

#bird options
bird_max_upward_angle = 35 #the maximum upward angle when flying up
bird_max_downward_angle = -90 #the maximum downward angle when flying down
bird_min_incremental_angle = 5 #the minimum incremental angle when tilting up or down
bird_angular_acceleration = 0.3 #the acceleration of bird's flying angle
bird_animation_time = 1 #the animation time of showing one image
bird_jump_velocity = -8 #the vertical jump up velocity
bird_acceleration = 3 #the gravity for the bird in the game
bird_max_displacement = 12 #the maximum displacement per frame
bird_starting_x_position = 150 #the starting x position of the bird
bird_starting_y_position = 250 #the starting y position of the bird
```

### Build A Bird


```python
#build the class Bird
class Bird:
    #Bird's attributes
    IMGS = BIRD_IMGS
    MAX_UPWARD_ANGLE = bird_max_upward_angle
    MAX_DOWNWARD_ANGLE = bird_max_downward_angle
    ANIMATION_TIME = bird_animation_time
    
    #initialize the Object
    def __init__(self, x_position, y_position):
        self.bird_img = self.IMGS[0] #use the first image as the initial image
        self.x = x_position #starting x position
        self.y = y_position #starting y position
        self.fly_angle = 0 #starting flying angle, initialized to be 0
        self.time = 0 #starting time set to calculate displacement, initialized to be 0
        self.velocity = 0 #starting vertical velocity, initialized to be 0
        self.animation_time_count = 0 #used to change bird images, initialized to be 0
        
    #defien a function to move the bird
    def move(self):
        self.time += 1 #count the time
        
        #for a body with a nonzero speed v and a constant acceleration a
        #the displacement d of this body after time t is d = vt + 1/2at^2
        displacement = self.velocity * self.time + (1/2) * bird_acceleration * self.time ** 2 #calculate the displacement when going downward
        
        #we don't want the bird going donw too fast
        #so we need to set a displacement limit per frame
        if displacement > bird_max_displacement:
            displacement = bird_max_displacement
        
        self.y = self.y + displacement #update the bird y position after the displacement
        
        if displacement < 0: #when the bird is going up
            if self.fly_angle < self.MAX_UPWARD_ANGLE: #if the flying angle is less than the maximum upward angle
                self.fly_angle += max(bird_angular_acceleration*(self.MAX_UPWARD_ANGLE - self.fly_angle), bird_min_incremental_angle) #accelerate the angle up
            else:
                self.fly_angle = self.MAX_UPWARD_ANGLE
                
        else: #when the bird is going down
            if self.fly_angle > self.MAX_DOWNWARD_ANGLE: #if the flying angle is less than the maximum downward angle
                self.fly_angle -= abs(min(bird_angular_acceleration*(self.MAX_DOWNWARD_ANGLE - self.fly_angle), -bird_min_incremental_angle)) #accelerate the angle down
            else:
                self.fly_angle = self.MAX_DOWNWARD_ANGLE

    #defien a function to jump the bird
    def jump(self):
        self.velocity = bird_jump_velocity #jump up by bird_jump_velocity
        self.time = 0 #when we jump, we reset the time to 0
    
    #define a function to get the rotated image and rotated rectangle for draw function
    def animation(self):
        self.animation_time_count += 1
        #if the bird is diving, then it shouldn't flap its wings        
        if self.fly_angle < -45:
            self.bird_img = self.IMGS[0]
            self.animation_time_count = 0 #reset the animation_time_count
        
        #if the bird is not diving, then it should flap its wings
        #keep looping the 3 bird images to mimic flapping its wings
        elif self.animation_time_count < self.ANIMATION_TIME:
            self.bird_img = self.IMGS[0]
        elif self.animation_time_count < self.ANIMATION_TIME * 2:
            self.bird_img = self.IMGS[1]
        elif self.animation_time_count < self.ANIMATION_TIME * 3:
            self.bird_img = self.IMGS[2]
        elif self.animation_time_count < self.ANIMATION_TIME * 4:
            self.bird_img = self.IMGS[1]
        else:
            self.bird_img = self.IMGS[0]
            self.animation_time_count = 0 #reset the animation_time_count
        
        #https://stackoverflow.com/questions/4183208/how-do-i-rotate-an-image-around-its-center-using-pygame
        #rotate the bird image for degree at self.tilt
        rotated_image = pygame.transform.rotate(self.bird_img, self.fly_angle)
        #store the center of the source image rectangle
        origin_img_center = self.bird_img.get_rect(topleft = (self.x, self.y)).center
        #update the center of the rotated image rectangle
        rotated_rect = rotated_image.get_rect(center = origin_img_center)
        #get the rotated bird image and the rotated rectangle
        return rotated_image, rotated_rect
```

### Build A Pipe


```python
#build the class Pipe
class Pipe:
    #Pipe's attributes
    VERTICAL_GAP = pipe_vertical_gap #the gap between the top and bottom pipes
    VELOCITY = pipe_velocity #the moving velocity of the pipes
    IMG_WIDTH = TOP_PIPE_IMG.get_width() #the width of the pipe
    IMG_LENGTH = TOP_PIPE_IMG.get_height() #the length of the pipe

    #initialize the Object
    def __init__(self, x_position):                
        self.top_pipe_img = TOP_PIPE_IMG #get the image for the pipe on the top
        self.bottom_pipe_img = BOTTOM_PIPE_IMG #get the image for the pipe on the bottom
        self.x = x_position #starting x position of the first set of pipes
        self.top_pipe_height = 0 #the height of the top pipe, initialized to be 0
        self.top_pipe_topleft = 0 #the topleft position of the top pipe, initialized to be 0
        self.bottom_pipe_topleft = 0 #the topleft position of the bottom pipe, initialized to be 0
        self.random_height() #set the height of the pipes randomly as well as the starting topleft position for top and bottom pipes
        
    #define a function to move the pipe
    def move(self):
        self.x -= self.VELOCITY
    
    #define a function to randomize pipe gaps
    def random_height(self):
        self.top_pipe_height = random.randrange(top_pipe_min_height, top_pipe_max_height) #the range is between top_pipe_min_height and top_pipe_max_height
        self.top_pipe_topleft = self.top_pipe_height - self.IMG_LENGTH #the topleft position of the top pipe should be the random height - the length of the pipe
        self.bottom_pipe_topleft = self.top_pipe_height + self.VERTICAL_GAP #the topleft position of the bottom pipe should be the random height + the GAP
```

### Build A Floor


```python
#build the class Floor
class Floor:
    #Floor's attributes
    IMGS = [FLOOR_IMG, FLOOR_IMG, FLOOR_IMG] #we need 3 floor images to set up the moving floor
    VELOCITY = floor_velocity #the moving velocity of the floor
    IMG_WIDTH = FLOOR_IMG.get_width() #the width of the floor

    #initialize the Object
    def __init__(self, y_position):
        #these 3 images have different starting position but have the same y position
        self.x1 = 0 #the starting x position of the first floor image
        self.x2 = self.IMG_WIDTH #the starting x position of the second floor image
        self.x3 = self.IMG_WIDTH * 2 #the starting x position of the third floor image
        self.y = y_position #the y position of the floor image
        
    #define a function to move the floor
    def move(self):
        self.x1 -= self.VELOCITY #move to the left with the velocity of VELOCITY
        self.x2 -= self.VELOCITY #move to the left with the velocity of VELOCITY
        self.x3 -= self.VELOCITY #move to the left with the velocity of VELOCITY
        
        if self.x1 + self.IMG_WIDTH < 0: #if the first floor image moves out of the screen 
            self.x1 = self.x3 + self.IMG_WIDTH #then move the first floor image to to the right of the third floor image
        if self.x2 + self.IMG_WIDTH < 0: #if the second floor image moves out of the screen 
            self.x2 = self.x1 + self.IMG_WIDTH #then move the second floor image to to the right of the first floor image
        if self.x3 + self.IMG_WIDTH < 0: #if the third floor image moves out of the screen 
            self.x3 = self.x2 + self.IMG_WIDTH #then move the third floor image to to the right of the second floor image
```

### Check Collision


```python
#define a function to check collision
def collide(bird, pipe, floor, screen):
    
    #Creates a Mask object from the given surface by setting all the opaque pixels and not setting the transparent pixels
    bird_mask = pygame.mask.from_surface(bird.bird_img) #get the mask of the bird
    top_pipe_mask = pygame.mask.from_surface(pipe.top_pipe_img) #get the mask of the pipe on the top
    bottom_pipe_mask = pygame.mask.from_surface(pipe.bottom_pipe_img) #get the mask of the pipe on the bottom
    
    sky_height = 0 #the sky height is the upper limit
    floor_height = floor.y #the floor height is the lower limit
    bird_lower_end = bird.y + bird.bird_img.get_height() #the y position of the lower end of the bird image
    
    #in order to check whether the bird hit the pipe, we need to find the point of intersection of the bird and the pipes
    #if overlap, then mask.overlap(othermask, offset) return (x, y)
    #if not overlap, then mask.overlap(othermask, offset) return None
    #more information regarding offset, https://www.pygame.org/docs/ref/mask.html#mask-offset-label
    top_pipe_offset = (round(pipe.x - bird.x), round(pipe.top_pipe_topleft - bird.y))
    bottom_pipe_offset = (round(pipe.x - bird.x), round(pipe.bottom_pipe_topleft - bird.y))
    
    #Returns the first point of intersection encountered between bird's mask and pipe's masks
    top_pipe_intersection_point = bird_mask.overlap(top_pipe_mask, top_pipe_offset)
    bottom_pipe_intersection_point = bird_mask.overlap(bottom_pipe_mask, bottom_pipe_offset)

    if top_pipe_intersection_point is not None or bottom_pipe_intersection_point is not None or bird_lower_end > floor_height or bird.y < sky_height:
        return True
    else:
        return False
```

### Draw Game Screen


```python
#define a function to draw the screen to display the game
def draw_game(screen, birds, pipes, floor, score, generation, game_time):
    
    #draw the background
    screen.blit(BG_IMG, (0, 0))
    
    #draw the moving floor
    screen.blit(floor.IMGS[0], (floor.x1, floor.y)) #draw the first floor image
    screen.blit(floor.IMGS[1], (floor.x2, floor.y)) #draw the second floor image
    screen.blit(floor.IMGS[2], (floor.x3, floor.y)) #draw the third floor image
    
    #draw the moving pipes
    for pipe in pipes:
        screen.blit(pipe.top_pipe_img, (pipe.x, pipe.top_pipe_topleft)) #draw the pipe on the top
        screen.blit(pipe.bottom_pipe_img, (pipe.x, pipe.bottom_pipe_topleft)) #draw the pipe on the bottom
    
    #draw the animated birds
    for bird in birds:
        rotated_image, rotated_rect = bird.animation()
        screen.blit(rotated_image, rotated_rect)
    
    #add additional information
    score_text = FONT.render('Score: ' + str(score), 1, FONT_COLOR) #set up the text to show the scores
    screen.blit(score_text, (SCREEN_WIDTH - 15 - score_text.get_width(), 15)) #draw the scores
    
    game_time_text = FONT.render('Timer: ' + str(game_time) + ' s', 1, FONT_COLOR) #set up the text to show the progress
    screen.blit(game_time_text, (SCREEN_WIDTH - 15 - game_time_text.get_width(), 15 + score_text.get_height())) #draw the progress
    
    generation_text = FONT.render('Generation: ' + str(generation - 1), 1, FONT_COLOR) #set up the text to show the number of generation
    screen.blit(generation_text, (15, 15)) #draw the generation
    
    bird_text = FONT.render('Birds Alive: ' + str(len(birds)), 1, FONT_COLOR) #set up the text to show the number of birds alive
    screen.blit(bird_text, (15, 15 + generation_text.get_height())) #draw the number of birds alive
    
    progress_text = FONT.render('Pipes Remained: ' + str(len(pipes) - score), 1, FONT_COLOR) #set up the text to show the progress
    screen.blit(progress_text, (15, 15 + generation_text.get_height() + bird_text.get_height())) #draw the progress
    
    pygame.display.update() #show the surface
```

### NEAT Parameters Setting


```python
#NEAT options
generation = 0 #note that the first generation of the birds is 0 because index starts from zero. XD
max_gen = 50 #the maximum number of generation to run
prob_threshold_to_jump = 0.8 #the probability threshold to activate the bird to jump
failed_punishment = 10 #the amount of fitness decrease after collision
```

### Configuration File


```python
#here is an example of the configuration file setting
#https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/config-feedforward
```

### Get Input Index


```python
#define a function to get the input index of the pipes
def get_index(pipes, birds):
    #get the birds' x position
    bird_x = birds[0].x
    #calculate the x distance between birds and each pipes
    list_distance = [pipe.x + pipe.IMG_WIDTH - bird_x for pipe in pipes]
    #get the index of the pipe that has the minimum non negative distance(the closest pipe in front of the bird)
    index = list_distance.index(min(i for i in list_distance if i >= 0)) 
    return index
```

### Visualization Functions

from https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/visualize.py


```python
import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()
    
def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
```

### Main Game Loop


```python
#import packages to build the AI
import neat

#define a function to run the main game loop
def main(genomes, config):
    
    global generation, SCREEN #use the global variable gen and SCREEN
    screen = SCREEN
    generation += 1 #update the generation
    
    score = 0 #initiate score to 0
    clock = pygame.time.Clock() #set up a clock object to help control the game framerate
    start_time = pygame.time.get_ticks() #reset the start_time after every time we update our generation
    
    floor = Floor(floor_starting_y_position) #build the floor
    pipes_list = [Pipe(pipe_starting_x_position + i * pipe_horizontal_gap) for i in range(pipe_max_num)] #build the pipes and seperate them by pipe_horizontal_gap
    
    models_list = [] #create an empty list to store all the training neural networks
    genomes_list = [] #create an empty list to store all the training genomes
    birds_list = [] #create an empty list to store all the training birds
    
    for genome_id, genome in genomes: #for each genome
        birds_list.append(Bird(bird_starting_x_position, bird_starting_y_position)) #create a bird and append the bird in the list
        genome.fitness = 0 #start with fitness of 0
        genomes_list.append(genome) #append the genome in the list
        model = neat.nn.FeedForwardNetwork.create(genome, config) #set up the neural network for each genome using the configuration we set
        models_list.append(model) #append the neural network in the list
        
    run = True
    
    while run is True: #when we run the program
        
        #check the event of the game and quit if we close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
        
        #stop the game when the score exceed the maximum score
        #break the loop and restart when no bird left
        if score >= max_score or len(birds_list) == 0:
            run = False
            break
        
        game_time = round((pygame.time.get_ticks() - start_time)/1000, 2) #record the game time for this generation
        
        clock.tick(FPS) #update the clock, run at FPS frames per second (FPS). This can be used to help limit the runtime speed of a game.
        
        floor.move() #move the floor
        
        pipe_input_index = get_index(pipes_list, birds_list) #get the input index of the pipes list
        
        passed_pipes = [] #create an empty list to hold all the passed pipes
        for pipe in pipes_list:
            pipe.move() #move the pipe
            if pipe.x + pipe.IMG_WIDTH < birds_list[0].x: #if the bird passed the pipe
                passed_pipes.append(pipe) #append the pipe to the passed pipes list
                       
        score = len(passed_pipes) #calculate the score of the game, which equals to the number of pipes the bird passed
        
        for index, bird in enumerate(birds_list):
            bird.move() #move the bird
            delta_x = bird.x - pipes_list[pipe_input_index].x #input 1: the horizontal distance between the bird and the pipe
            delta_y_top = bird.y - pipes_list[pipe_input_index].top_pipe_height #input 2: the vertical distance between the bird and the top pipe
            delta_y_bottom = bird.y - pipes_list[pipe_input_index].bottom_pipe_topleft #input 3: the vertical distance between the bird and the bottom pipe
            net_input = (delta_x, delta_y_top, delta_y_bottom)
            #input the bird's distance from the pipes to get the output of whether to jump or not
            output = models_list[index].activate(net_input)
            
            if output[0] > prob_threshold_to_jump: #if the model output is greater than the probability threshold to jump
                bird.jump() #then jump the bird
            
            bird_failed = True if collide(bird, pipes_list[pipe_input_index], floor, screen) is True else False
            
            #the fitness function is a combination of game score, alive time, and a punishment for collision
            genomes_list[index].fitness = game_time + score - bird_failed * failed_punishment
            
            if bird_failed:
                models_list.pop(index) #drop the model from the list if collided
                genomes_list.pop(index) #drop the genome from the list if collided
                birds_list.pop(index) #drop the bird from the list if collided

        draw_game(screen, birds_list, pipes_list, floor, score, generation, game_time) #draw the screen of the game
```

### Run NEAT


```python
#define a function to run NEAT algorithm to play flappy bird
def run_NEAT(config_file):
    
    #the template for the configuration file can be found here:
    #https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/config-feedforward
    #the description of the options in the configuration file can be found here:
    #https://neat-python.readthedocs.io/en/latest/config_file.html#defaultgenome-section
    
    #use NEAT algorithm to build a neural network based on the pre-set configurtion
    #Create a neat.config.Config object from the configuration file
    config = neat.config.Config(neat.DefaultGenome, 
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, 
                                neat.DefaultStagnation,
                                config_file)
    
    #Create a neat.population.Population object using the Config object created above
    neat_pop = neat.population.Population(config)
    
    #show the summary statistics of the learning progress
    neat_pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    neat_pop.add_reporter(stats)
    
    #Call the run method on the Population object, giving it your fitness function and (optionally) the maximum number of generations you want NEAT to run
    neat_pop.run(main, max_gen)
    
    #get the most fit genome genome as our winner with the statistics.best_genome() function
    winner = stats.best_genome()
    
    #visualize the results
    node_names = {-1:'delta_x', -2: 'delta_y_top', -3:'delta_y_bottom', 0:'Jump or Not'}
    draw_net(config, winner, True, node_names = node_names)
    plot_stats(stats, ylog = False, view = True)
    plot_species(stats, view = True)
    
    #show the final statistics
    print('\nBest genome:\n{!s}'.format(winner))
```

### Run Flappy Bird


```python
#run the game!
config_file = 'config-feedforward.txt'
run_NEAT(config_file)
```
