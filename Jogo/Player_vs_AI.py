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
    checkpoint_file = "model.checkpoint.zip"

    # Join the path components
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)


    config=config_mu4x4.MuZeroConfig()
    
elif jogo=="A5x5":
    board = Board(5, 5)
    
    # Load Model
    checkpoint_folder = r"logs\ataxx_5x5\2024-01-12--09-36-05"
    checkpoint_file = "model.checkpoint.zip"
    
    # Join the path components    
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
    
    config=config_mu5x5.MuZeroConfig()
  
elif jogo=="A6x6":
    board = Board(6, 6)
    
    # Load Model    
    checkpoint_folder = r"logs\ataxx_6x6\2024-01-11--19-39-14"
    checkpoint_file = "model.checkpoint.zip"
    
    # Join the path components    
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
    config=config_mu6x6.MuZeroConfig()



FPS = 60

run = True
clock = pygame.time.Clock()

selected_piece_position = None
selected_piece_future_position = None

player_turn_index = 1
ai_player_index = -1  # -1- Muzero play second 1- play first

model=board.load_model(checkpoint_path)
selfplay=selffplay(model,numpy.random.randint(10000),board,config)

board.DrawBoard()
pygame.display.update()
time.sleep(2)

while run:
    clock.tick(FPS)

    if player_turn_index == ai_player_index:
        selfplay.play_game(0,0,ai_player_index)
        
        
        player_turn_index *= -1
        run = board.CheckGame()
    else:
        
        for event in pygame.event.get():
                
            if event.type == pygame.MOUSEBUTTONDOWN and selected_piece_future_position is None and selected_piece_position is not None:
                selected_piece_future_position = board.ConvertToBoardValues(pygame.mouse.get_pos())
                board.clear_pos()
                piecevalue = board.PieceAt(selected_piece_future_position)
                distanceX, distanceY = board.GetDistanceBoardUnits(selected_piece_position, selected_piece_future_position)
                if (distanceX == 1 and distanceY <= 1) or (distanceY == 1 and distanceX <= 1):
                    if piecevalue == 0:
                        board.MakeNewPieceAt(selected_piece_future_position, player_turn_index)
                        board.CatchPiece(selected_piece_future_position, player_turn_index)
                       
                        player_turn_index *= -1
                elif (distanceX == 2 and distanceY <= 2) or (distanceY == 2 and distanceX <= 2):
                    if piecevalue == 0:
                        board.MovePieceTo(selected_piece_position, selected_piece_future_position, player_turn_index)
                        board.CatchPiece(selected_piece_future_position, player_turn_index)
                        player_turn_index *= -1
                run = board.CheckGame()
                
                selected_piece_position = None
                selected_piece_future_position = None
            elif event.type == pygame.MOUSEBUTTONDOWN and selected_piece_position is None:
                
                board.clear_pos()
                selected_piece_position = board.ConvertToBoardValues(pygame.mouse.get_pos())
                board.possibles(selected_piece_position,player_turn_index)
                x, y = selected_piece_position
                if board.BoardMatrix[x][y] != player_turn_index:
                    selected_piece_position = None

    board.DrawBoard()
    pygame.display.update()
    
board.end()
pygame.display.update()
time.sleep(5)

