import time
import numpy as np
import pyopencl as cl
import pygame
from pygame.locals import *
import pyopencl.array as cl_array
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from time import clock

pygame.init()

screen = pygame.display.set_mode((128, 128))
striped = np.zeros((128, 128))
striped.fill((200 + (200*256) + (200*256*256)))

print(striped)

#screen.set_at((100, 100), 0xFFFFFF)

pygame.surfarray.blit_array(screen, striped)
    
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    
        pygame.display.flip()
    pygame.time.wait(10)
