from tkinter import *
import numpy as np
import customtkinter
import random
import copy

from self_play import GameHistory, MCTS 
import models
import torch
from muzero2 import CPUActor
import ray
import os
import pathlib
import time

import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(True)


def passar_turno(): #Botao para passar o turno
    global turn, passe
    turn = go.change(turn)
    passe +=1
    if passe ==2:
        go.game_over(turn)

def tamanho9(): #Definir o tamanho do tabuleiro
    global tam
    tam = 9
    menu()

def tamanho7(): #Definir o tamanho do tabuleiro
    global tam
    tam = 7
    menu()
    
def menu():  #Menu inicial do jogo (parte grafica)
    global go, frame_cast, frame_score, frame_score2, frame_pecasb, frame_pecasw, tam_quadrado, simb_espe, canvas, canvas2, canvas3, button_passar
    
    tam_quadrado = 490/(tam-1) 

    simb_espe=tam_quadrado/2  # Tamanho do símbolo

    button_13.destroy()
    button_9.destroy()
    Label_principal.destroy()

    # Adicionando um botão para passar o turno
    button_passar = customtkinter.CTkButton(frame,text="Passar Turno", command=passar_turno)
    button_passar.place(x=950 ,y=20)


    #Tabuleiro
    frame_cast = customtkinter.CTkFrame(master=frame,fg_color="#DAA520")
    frame_cast.pack(pady=70, padx=250, fill="both", expand=True)

    frame_score = customtkinter.CTkFrame(frame, width=140, height=200)
    frame_score.place(x=950 ,y=90)

    frame_score2 = customtkinter.CTkFrame(frame, width=140, height=200)
    frame_score2.place(x=50 ,y=450)

    frame_pecasb = customtkinter.CTkFrame(frame, width=140, height=140,border_width=4, fg_color="white", border_color="#DAA520")
    frame_pecasb.place(x=950 ,y=450)

    frame_pecasw = customtkinter.CTkFrame(frame, width=140, height=140,border_width=4, fg_color="white", border_color="#DAA520")
    frame_pecasw.place(x=50 ,y=50)

    canvas = customtkinter.CTkCanvas(frame_cast,
                                bg="#DAA520")

    canvas2 = customtkinter.CTkCanvas(frame_pecasb,width=140, height=140,
                                    bg="#DAA520")
    canvas2.pack()

    canvas3 = customtkinter.CTkCanvas(frame_pecasw,width=140, height=140,
                                    bg="#DAA520")
    canvas3.pack()

    go = GO()
    #go.game_geral()

def menu_fim(black_score, white_score):  #Menu final do jogo (parte grafica)
    
    frame_cast.destroy()
    frame_pecasb.destroy()
    frame_pecasw.destroy()
    button_passar.destroy()

    go.desenhar_whiteblack()
    
    Label_over = customtkinter.CTkLabel(frame, text="Game Over", font=("Helvetica", 30))
    Label_over.pack(padx=10,pady=(200,20))

    if black_score > white_score:
        Label_black = customtkinter.CTkLabel(frame, text="Black Win", font=("Helvetica", 60))
        Label_black.pack(padx=10,pady=(20,20))

    elif black_score < white_score:
        Label_white = customtkinter.CTkLabel(frame, text="White Win", font=("Helvetica", 60))
        Label_white.pack(padx=10,pady=(20,20))

    else:
        Label_empate = customtkinter.CTkLabel(frame, text="Draw", font=("Helvetica", 60))
        Label_empate.pack(padx=10,pady=(20,20))



global turn
turn = 1  # pretas=1     brancas=2

NORTH = (-1,0)
SOUTH = (1,0)
EAST  = (0,1)
WEST  = (0,-1)

#  L  | C O L U N A S (y)
#  I  |
#  N  |
#  H  | 
#  A  |
#  S  |
# (x) |



class GO:
    def __init__(self):
        self.board = np.zeros(shape=(tam, tam))
        self.coor1 = np.array((20,20))
        self.coor2 = np.array((20,20))
        self.last = np.zeros(shape=(tam, tam)) #Para verificar ko
        self.score_black = 0
        self.score_white = 0
        self.desenhar_tabuleiro()
        self.c = 0
        root.bind('<Button-1>', self.game_geral)  # ##########
        self.game_is_over=False
        self.passe = 0
        self.label_score_black2
        self.label_score_white2
        self.label_pecas_black2
        self.label_pecas_white2
        self.label_pecas_white
        self.label_pecas_black
        if tam==7:
            import games.go as go_muzero
            checkpoint_folder = r"logs\go\2024-01-12--14-12-21"  #Carregar o modelo ia
        else:
            import games.go9 as go_muzero
            checkpoint_folder = r"logs\go9\2024-01-11--17-05-02"  #Carregar o modelo ia
        checkpoint_file = "model.checkpoint"
        self.game = go_muzero.Game()
        self.config = go_muzero.MuZeroConfig()   
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        initial_checkpoint = self.load_model(checkpoint_path)
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()
        self.game_history = GameHistory()
        self.observation = self.game.reset()


        
    def desenhar_tabuleiro(self): #Desenho do tabuleiro de jogo
        #canva.delete("all")

        # Desenhar tabuleiro
        for i in range(tam):
            canvas.create_line(i * tam_quadrado+70, 40, i * tam_quadrado+70, 570-40)
            canvas.create_line(70, i * tam_quadrado+40, 630-70, i * tam_quadrado+40)
        #canvas.create_oval(313, 283, 317, 287, fill="black")
        i=tam-1
        canvas.create_line(70, i * tam_quadrado+40, 631-70, i * tam_quadrado+40)
        canvas.pack(fill=BOTH, expand=YES)

        print(self.board)
        if(tam==9):
            self.Black=41
            self.White=40
        else:
            self.Black=25
            self.White=24

        canvas.create_aa_circle(
                600,
                550,
                10,
                simb_espe,
                fill='black'
            )
        # Score
        label_score_black = customtkinter.CTkLabel(frame_score, text="Black Score", font=("Arial",20))
        label_score_black.pack(padx=16,pady = 20)

        self.label_score_black2 = customtkinter.CTkLabel(frame_score, text=self.score_black, font=("Arial",15))
        self.label_score_black2.pack(padx=10,pady=10)

        label_score_white = customtkinter.CTkLabel(frame_score2, text="White Score", font=("Arial",20))
        label_score_white.pack(padx=16,pady = 20)

        self.label_score_white2 = customtkinter.CTkLabel(frame_score2, text=self.score_white, font=("Arial",15))
        self.label_score_white2.pack(padx=10,pady=10)
        
        #Pecas
        self.label_pecas_black = customtkinter.CTkLabel(frame_score, text="Black Pieces", font=("Arial",20))
        self.label_pecas_black.pack(padx=16,pady = 20)

        self.label_pecas_black2 = customtkinter.CTkLabel(frame_score, text=self.Black, font=("Arial",15))
        self.label_pecas_black2.pack(padx=10,pady=10)

        self.label_pecas_white = customtkinter.CTkLabel(frame_score2, text="White Pieces", font=("Arial",20))
        self.label_pecas_white.pack(padx=16,pady = 20)

        self.label_pecas_white2 = customtkinter.CTkLabel(frame_score2, text=self.White, font=("Arial",15))
        self.label_pecas_white2.pack(padx=10,pady=10)
        

    def set_peca(self, coor,turn): #coloca a peça no tabuleiro
        if turn==1:
            self.board[coor[0]][coor[1]] = 1
        else:
            self.board[coor[0]][coor[1]] = 2
        

    def desenhar_peca(self, coor,turn): #Desenha a peça no tabuleiro
        if turn==1:
            i = coor[0]
            j = coor[1]
            canvas.create_aa_circle(
                i * tam_quadrado +71,
                j * tam_quadrado +43,
                int(simb_espe),
                simb_espe,
                fill='black'
            )
        else:
            i = coor[0]
            j = coor[1]
            canvas.create_aa_circle(
                i * tam_quadrado +71,
                j * tam_quadrado +43,
                int(simb_espe),
                simb_espe,
                fill='white'
            )

    def reset_peca(self, coor, turn): #Faz um reset dos  valores iniciais
        i = coor[0]
        j = coor[1]
        obj_id = canvas.find_overlapping(i * tam_quadrado  +71,
                j * tam_quadrado +43,i * tam_quadrado +71,
                j * tam_quadrado +43)
        canvas.delete(obj_id)
        if turn ==2:
            canvas2.create_aa_circle(
                random.randint(5, 95),
                random.randint(5, 95),
                10,
                simb_espe,
                fill='white'
            )
        else:
            canvas3.create_aa_circle(
                random.randint(4, 96),
                random.randint(4, 96),
                10,
                simb_espe,
                fill='black'
            )
                

    def jogada(self, coor):  #Realiza uma jogada
        global turn
        self.visited = [coor]

        self.set_peca(coor, turn)
        if not self.check_ko(coor): 
            if turn == 1:
                self.coor1=[coor]
            else:
                self.coor2=[coor]
            
            if self.legal_move(coor, turn) and self.check_pecas(turn):
                self.desenhar_peca(coor, turn)
                turn = self.change(turn)
                self.check_cap(coor, turn)    #como a turn ja foi mudada envio so a turn pq quero ver quantas peças adversarias ficam rodeadas
                
                
            elif self.check_cap(coor, self.change(turn)):  
                #print("cap")
                self.desenhar_peca(coor, turn)
                turn = self.change(turn)
            
            #else:
            #    self.board[coor[0]][coor[1]]=0
            #    print("Jogada inválida")
            #    return False

            if turn == 2:
                self.Black -= 1
            else:
                self.White -= 1
                
            if(self.Black==0 and turn==1):
                #print("ficou sem peças o jogador Preto")
                #turn=self.change(turn)
                flag1=True

            elif(self.White==0 and turn==2):
                #print("ficou sem peças o jogador Branco")
                #turn=self.change(turn)
                flag2= True
            #if(flag1 and flag2):
            self.game_over(turn)
            return True 
        return False
            # A mensagem na interface e outras operações após uma jogada válida
            

    def check_ko(self,coor):  #Verifica se ocorre uma jogada ko
        if np.all(self.last == np.zeros(shape=(tam, tam))):
            self.last = copy.deepcopy(self.board)
            return False
        if np.all(self.last == self.board):
            print("Repetição de estado (ko). Jogada inválida.")
            self.board[coor[0]][coor[1]]=0
            return True
        self.last = copy.deepcopy(self.board)
        return False

    def game_over(self,turn): #Verifica se o jogo acabou
        global passe
        if(passe>=2 or (self.Black==0 and self.White==0)):
            self.score()
            self.game_is_over=True
            # Desativa o botão após o jogo acabar
            #botao_passar_turno.config(state=DISABLED)
            menu_fim(self.score_black, self.score_white)

            
    def score(self): #Verifica o score no final do jogo e da print dele
        self.territorio()
        self.score_white=5.5+self.score_white
        self.score_black=0+self.score_black
        print(self.score_black)
        print(self.score_white)
        if(self.score_white>self.score_black):
            print("Brancas Ganham")
        elif(self.score_white<self.score_black):
            print("Pretas Ganham")
        else:
            print("Empate")

    
        ####-----------------------------------------Território---------------------------------------------####
    def lib_territorios(self,coor):
        neigh=[]
           
        for element in [EAST,SOUTH,WEST,NORTH]:
            pos=(coor[0]+element[0],coor[1]+element[1])
            if(0<=pos[0]<tam and 0<=pos[1]<tam):
                neigh.append(pos)
        #print(neigh)
        return neigh
    
    def territorio(self):
        self.visited=[]
        for i in range(tam):
            for j in range(tam):
                if(self.board[i][j]==0):
                    neigh=self.lib_territorios((i,j))
                    if(not any(np.array_equal((i,j), item) for item in self.visited)):
                        self.visited.append((i,j))
                        self.cont_terr(neigh)
                    
    def cont_terr(self,neigh):
        stack=list()
        c=1
        flagBlack=False
        flagWhite=False
        for space in neigh:
            stack.append(space)
            while(stack):
                stone=stack.pop()
                
                if(self.board[stone[0]][stone[1]]==0 and not any(np.array_equal(stone, item) for item in self.visited)):
                    neigh=self.lib_territorios(stone)
                    c+=1
                    self.visited.append(stone)
                    stack.extend(neigh)       # adicionar os vizinhos todos à lista
                elif(self.board[stone[0]][stone[1]]==1):
                    flagBlack=True
                elif(self.board[stone[0]][stone[1]]==2):
                    flagWhite=True
                    
        if(flagWhite and flagBlack):
            return 
        elif(flagBlack):
            self.score_black+=c
        elif(flagWhite):
            self.score_white+=c
            
      ####-----------------------------------------------------------------------------------------####
    
    def check_pecas(self,turn): #Verifica se uma peça é do jogador atual
        if(turn==1 and self.Black>0):
            return True
        if(turn==2 and self.White>0):
            return True
        return False

                
    def check_cap(self,coor,turn):  #Verifica se uma peça foi capturada 
        
        _,neigh=self.liberties(coor,turn)
        for stone in neigh:
            self.cap=[]
            if (self.board[stone[0]][stone[1]]== turn):
                if(self.capture(stone,turn)):
                    #print(self.cap)
                    for pedra in self.cap:
                        self.board[pedra[0]][pedra[1]]=0
                        self.reset_peca(pedra,turn)
                        if turn ==1:
                            self.score_white+=1
                        else:
                            self.score_black+=1
                    return True

        return False
                
            
    def capture(self,stone,turn):  #Verifica as possiveis peças que podem ser capturadas
        stack=list()
        stack.append(stone)
        #print(stack)   # adicionar o primeiro elemento
        while(stack):
            stone=stack.pop()
            
            if(self.board[stone[0]][stone[1]]==turn and not any(np.array_equal(stone, item) for item in self.cap)):
                val,neigh=self.liberties(stone,turn)
                if val:
                    self.cap=[]
                    return False
                self.cap.append(stone)
                stack.extend(neigh)       # adicionar os vizinhos todos à lista
                #print(neigh)
        return True
    
    
    
    def legal_move(self,coor,turn):  #Verifica se o é possivel colocar uma peça naquele local
        
        val,neigh=self.liberties(coor,turn)
        if val:
            return True

        self.set_peca(coor,turn)  #como chamo a recursivamente esta funçao no check_color tenho que  ter isto aqui
        
        if(self.check_color(neigh,turn)):
            return True
        
        
        return False
    
    def check_color(self,neigh,turn):  #Verifica se peças adjacentes são da mesma cor ou não
        
        for stone in neigh:
            
            
            if (self.board[stone[0]][stone[1]]== turn and not any(np.array_equal(stone, item) for item in self.visited)):  # é criado uma variavel item que percorre o self.visited e ve se exsite algum valor igual à
                #print("entrou" +str(stone))
                #print(self.board[stone[0]][stone[1]])
                #print(self.board)
                self.visited.append(stone)
                if (self.legal_move(stone, turn)):
                    return True
                  
        return False
              

    def liberties(self,coor,turn): # verifica se existem liberdades e os vizinhos da mesma cor
        neigh=[]
        c=0
        for element in [EAST,SOUTH,WEST,NORTH]:
            pos=(coor[0]+element[0],coor[1]+element[1])
            if(0<=pos[0]<tam and 0<=pos[1]<tam):
                #neigh.append(pos)
                if(self.board[pos[0]][pos[1]]==0):
                    c+=1
                elif(self.board[pos[0]][pos[1]]==1 and turn==1):
                    #print("entrou 1")
                    neigh.append(pos)
                elif(self.board[pos[0]][pos[1]]==2 and turn==2):
                    #print("entrou 2")
                    neigh.append(pos)
                    

        if c>0:
            return True,neigh
        
        return False,neigh

    def convert_pixels_to_coords(self, grid_pos):  #Recebe a coordenada clicada em pixels e converte-a na posição respetiva do tabuleiro ( 0,0 ) -> (nb-1,nb-1)
        #print(grid_pos)
        grid_pos = np.array(grid_pos)
        n = int((grid_pos[0]-30) // tam_quadrado),int((grid_pos[1]-20) // tam_quadrado)
        return np.array(n)
        #return np.array(grid_pos // tam_quadrado, dtype=int)
    

    def change(self, turn):  #Remove uma peça do tabuleiro
        if turn == 1:
            canvas.create_aa_circle(
                22,
                22,
                10,
                simb_espe,
                fill='white'
            )
            obj_id = canvas.find_overlapping(600,550,600,550)
            canvas.delete(obj_id)
            return 2
        else:
            canvas.create_aa_circle(
                600,
                550,
                10,
                simb_espe,
                fill='black'
            )
            obj_id = canvas.find_overlapping(22,22,22,22)
            canvas.delete(obj_id)
            return 1

    def click(self, event): # Evento principal do jogo quando o jogo é human vs human, quando clique na parte grafica verifica se pode jogar na posiçao clicada e se possivel joga
        global passe
        flag=0
        if not self.game_is_over:
            if event.x < 600 and event.y < 560 and event.x >=40 and event.y >= 15:
                pixels_position = [event.x, event.y]
                coords_pos = self.convert_pixels_to_coords(pixels_position)
                if self.pos_clear(coords_pos):
                    t=self.jogada(coords_pos)
                    self.desenhar_whiteblack()
                    if(t):
                        # arranjar esta parte
                        print(self.board)
                        print("Black Score: " + str(self.score_black))
                        print("Black Stones: " + str(self.Black))
                        print("White Score: " + str(self.score_white))
                        print("White Stones: " + str(self.White))
                        passe = 0

    def game_geral(self, event): #Evento principal do jogo, computador vs computador 
        global passe
        global turn

        temperature = 0
        temperature_threshold = 0
        #opponent = "human"
        muzero_player = 1 #self.config.muzero_player 

        if self.c == 0:       
            self.game_history.action_history.append(0)
            self.game_history.observation_history.append(self.observation)
            self.game_history.reward_history.append(0)
            self.game_history.to_play_history.append(self.game.to_play())
            self.c = 1

        done = False
        #while (not done and len(self.game_history.action_history) <= self.config.max_moves):
        
        '''
        assert (
            len(np.array(self.observation).shape) == 3
        ), f"Observation should be 3 dimensionnal instead of {len(np.array(self.observation).shape)} dimensionnal. Got observation of shape: {np.array(observation).shape}"
        assert (
            np.array(observation).shape == self.config.observation_shape
        ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {np.array(observation).shape}."
        '''
        stacked_observations = self.game_history.get_stacked_observations(
            -1, self.config.stacked_observations, len(self.config.action_space)
        )

        # Choose the action
        if muzero_player == self.game.to_play(): 
            #time.sleep(1)
            #print(self.config.action_space)
            root1, mcts_info = MCTS(self.config).run(
                self.model, 
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )

            action = self.select_action( 
                root1,
                temperature
                if not temperature_threshold
                or len(self.game_history.action_history) < temperature_threshold
                else 0,
            )
            #Faz print no tabuleiro 
            if action == tam*tam:
                passar_turno()
                print('Muzero passou o turno')
            else:
                for x in range(tam):
                    for y in range(tam):
                        if x * tam + y == action:
                            coords_pos = (x,y)

                t=self.jogada(coords_pos)
                self.desenhar_whiteblack()
                if(t):
                    # arranjar esta parte
                    print(self.board)
                    print("Black Score: " + str(self.score_black))
                    print("Black Stones: " + str(self.Black))
                    print("White Score: " + str(self.score_white))
                    print("White Stones: " + str(self.White))
                    passe = 0
            
            observation, reward, done = self.game.step(action)


            self.game_history.store_search_statistics(root1, self.config.action_space)

            # Next batch
            self.game_history.action_history.append(action)
            self.game_history.observation_history.append(observation)
            self.game_history.reward_history.append(reward)
            self.game_history.to_play_history.append(self.game.to_play())

            #Enviar jogada

        else:
            #time.sleep(1)
            #Reecebe jogada random no momento
            action = self.expert_action(2)

            #Coloca jogada nos dois tabuleiros
            #action = coords_pos[0]*7+coords_pos[1]
            
            #Faz print no tabuleiro 
            if action == tam*tam:
                passar_turno()
                print('Muzero passou o turno')
            else:
                for x in range(tam):
                    for y in range(tam):
                        if x * tam + y == action:
                            coords_pos = (x,y)

                t=self.jogada(coords_pos)
                self.desenhar_whiteblack()
                if(t):
                    # arranjar esta parte
                    print(self.board)
                    print("Black Score: " + str(self.score_black))
                    print("Black Stones: " + str(self.Black))
                    print("White Score: " + str(self.score_white))
                    print("White Stones: " + str(self.White))
                    passe = 0
            
            observation, reward, done = self.game.step(action)


            #self.game_history.store_search_statistics(root1, self.config.action_space)

            # Next batch
            self.game_history.action_history.append(action)
            self.game_history.observation_history.append(observation)
            self.game_history.reward_history.append(reward)
            self.game_history.to_play_history.append(self.game.to_play())


        #return game_history
                    
    def expert_action(self, turn):  #Jogada ramdom
        action = np.random.choice(self.legal_actions(turn))
        return action
    
    def legal_actions(self, turn): #Movimentos possiveis
        legal = []

        for i in range(tam):
            for j in range(tam):
                self.visited = [(i,j)]
                board_new = np.copy(self.board)
                if self.confirm_jog(board_new, (i,j), turn):
                    legal.append(i*tam+j)
        legal.append(tam*tam) #jogada de passar o turno
        return legal
    
    def confirm_jog(self,board_new, coor, turn):     #Confirma se é uma jogada possivel       
        #Verificar se local é 0
        if board_new[coor[0]][coor[1]]!=0:
            return False
        
        #Efetuar jogada na copia do tabuleiro
        if turn==1:
            board_new[coor[0]][coor[1]] = 1
        else:
            board_new[coor[0]][coor[1]] = 2

        #Verificar check_ko
        if not np.all(self.last == np.zeros(shape=(tam, tam))):
            if np.all(self.last == board_new):
                return False

        #Verificar legal_move
        if not self.legal_move2(coor,board_new, turn):
            return False
        
        #Verificar check_pecas 
        if turn==1:
            if not self.Black>0:
                return False
        if turn==2:
            if not self.White>0:
                return False 
        
        return True  
    
    def legal_move2(self,coor,board_n, turn): #Verifica se o é possivel colocar uma peça naquele local
            
        val,neigh=self.liberties_legal(coor,board_n, turn)
        if val:
            return True

        if turn==1:
            board_n[coor[0]][coor[1]] = 1
        else:
            board_n[coor[0]][coor[1]] = 2
        
        if(self.check_color2(neigh, board_n, turn)):
            return True
        
        return False
    
    def liberties_legal(self,coor, board_n, turn): # verifica se existem liberdades e os vizinhos da mesma cor
        neigh=[]
        c=0
        for element in [EAST,SOUTH,WEST,NORTH]:
            pos=(coor[0]+element[0],coor[1]+element[1])
            if(0<=pos[0]<tam and 0<=pos[1]<tam):
                #neigh.append(pos)
                if(board_n[pos[0]][pos[1]]==0):
                    c+=1
                elif(board_n[pos[0]][pos[1]]==1 and turn==1):
                    #print("entrou 1")
                    neigh.append(pos)
                elif(board_n[pos[0]][pos[1]]==2 and turn==2):
                    #print("entrou 2")
                    neigh.append(pos)
                    

        if c>0:
            return True,neigh
        
        return False,neigh
    
    def check_color2(self,neigh, board_n, turn): #Verifica se peças adjacentes são da mesma cor ou não
        
        for stone in neigh:
            
            
            if (board_n[stone[0]][stone[1]]== turn and not any(np.array_equal(stone, item) for item in self.visited)):  # é criado uma variavel item que percorre o self.visited e ve se exsite algum valor igual à
                #print("entrou" +str(stone))
                #print(self.board[stone[0]][stone[1]])
                #print(self.board)
                self.visited.append(stone)
                if (self.legal_move2(stone,board_n,turn)):
                    return True
                  
        return False
                
    def select_action(self, node, temperature):  #Seleciona uma ação por base na rede neuronal
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32" 
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)
        #print("açao")
        #print(action)
        #print(board)
        return action


    def desenhar_whiteblack(self):  #Parte grafica para a peças brancas e pretas(peças/peças capturadas/score)
        self.label_score_black2.destroy()
        self.label_score_white2.destroy()
        self.label_pecas_white.destroy()
        self.label_pecas_black.destroy()
        self.label_score_white2 = customtkinter.CTkLabel(frame_score2, text=self.score_white, font=("Arial",15))
        self.label_score_white2.pack(padx=10,pady=10)
        self.label_score_black2 = customtkinter.CTkLabel(frame_score, text=self.score_black, font=("Arial",15))
        self.label_score_black2.pack(padx=10,pady=10)
        self.label_pecas_black2.destroy()
        self.label_pecas_white2.destroy()
        self.label_pecas_white = customtkinter.CTkLabel(frame_score2, text="White Pieces", font=("Arial",20))
        self.label_pecas_black = customtkinter.CTkLabel(frame_score, text="Black Pieces", font=("Arial",20))
        self.label_pecas_white.pack(padx=16,pady = 20)
        self.label_pecas_black.pack(padx=16,pady = 20)
        self.label_pecas_white2 = customtkinter.CTkLabel(frame_score2, text=self.White, font=("Arial",15))
        self.label_pecas_white2.pack(padx=10,pady=10)
        self.label_pecas_black2 = customtkinter.CTkLabel(frame_score, text=self.Black, font=("Arial",15))
        self.label_pecas_black2.pack(padx=10,pady=10)
                        

    def pos_clear(self, coor):  #Verifica se a posição esta livre
        return self.board[coor[0]][coor[1]] == 0
    
    def load_model(self, checkpoint_path): #Faz load do modelo
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path)
            #print(f"\nUsing checkpoint from {checkpoint_path}")
            
        #print("model loaded")
        return self.checkpoint


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

#Inicio da parte grafica

root = customtkinter.CTk()
root.title("GO")
root.geometry("1200x750")


global passe
passe = 0

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=40, fill="both", expand=True)

Label_principal = customtkinter.CTkLabel(frame, text="Escolha o tamanho do Mapa", font=("Helvetica", 30))
Label_principal.pack(padx=10,pady=(100,20))

button_9 = customtkinter.CTkButton(frame,width=200 ,height=50,text="7", command=tamanho7)
button_9.pack(padx=20,pady=20)

button_13 = customtkinter.CTkButton(frame,width=200 ,height=50,text="9", command=tamanho9)
button_13.pack(padx=10,pady=10)


root.mainloop()

# arranjar os botoes, ao carregar no botao ele coloca uma peça

