import pygame
from Board import Board
from time import sleep
import os
import numpy
from Board import selffplay
import config_mu4x4
import config_mu5x5
import config_mu6x6
import time

print("Modo de jogo: A4x4, A5x5, A6x6")
jogo=input()
if jogo == "A4x4":
    board = Board(4, 4)
    
    # Load Model
    checkpoint_folder = r"logs\ataxx_4x4\2024-01-12--14-33-13"
    checkpoint_file = "model.checkpoint"

    # Join the path components
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)

    
    config=config_mu4x4.MuZeroConfig()
    
elif jogo=="A5x5":
    board = Board(5, 5)

    # Load Model
    checkpoint_folder = r"logs\ataxx_5x5\2024-01-12--09-36-05"
    checkpoint_file = "model.checkpoint"
    
    # Join the path components    
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)

    config=config_mu5x5.MuZeroConfig()
  
elif jogo=="A6x6":
    board = Board(6, 6)
    
    # Load Model    
    checkpoint_folder = r"logs\ataxx_6x6\2024-01-11--19-39-14"
    checkpoint_file = "model.checkpoint"
    
    # Join the path components    
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)

    
    config=config_mu6x6.MuZeroConfig()



FPS = 60
run = True
clock = pygame.time.Clock()

selected_piece_position = None
selected_piece_future_position = None

player_turn_index = 1
model=board.load_model(checkpoint_path)


selfplay=selffplay(model,numpy.random.randint(10000),board,config)
selfplay2=selffplay(model,numpy.random.randint(10000),board,config)

board.DrawBoard()
pygame.display.update()
time.sleep(2)
while run:
    
    clock.tick(FPS)

    if player_turn_index == -1:
        time.sleep(1)
        
        selfplay.play_game(0,0,player_turn_index)
        

        player_turn_index *= -1
        run = board.CheckGame()
        
    else:
        time.sleep(1)
        selfplay2.play_game(0,0,player_turn_index)
        
        
        
        player_turn_index *= -1
        run = board.CheckGame()
        
        

    board.DrawBoard()
    pygame.display.update()
    
board.end()
pygame.display.update()
time.sleep(5)
