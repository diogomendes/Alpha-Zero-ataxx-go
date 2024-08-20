import numpy as np
import datetime
import pathlib
import torch
from .abstract_game import AbstractGame
import copy
import pickle

tam = 9
NORTH = (-1,0)
SOUTH = (1,0)
EAST  = (0,1)
WEST  = (0,-1)

# change values to find the best parameters
class MuZeroConfig:
    def __init__(self):
        self.seed=0
        self.max_num_gpus=None
        
        ### Game
        self.observation_shape=(3,tam,tam) # Dimensions (number of diferent values (player1,player2, empty), height,width)
        self.action_space=list(range((tam*tam)+1)) #all possible actions e +1 (jogada de passar o turno), vai ficar com o ultimo valor, em 7x7(49)
        #self.action_space=[(x, y) for x in range(tam) for y in range(tam)]
        #self.action_space.append((-1,-1)) #passar
        self.players=list(range(2))  # list of players
        self.stacked_observations=0  # Number of previous observations and previous actions to add to the current observation
        
        # evalute
        self.muzero_player =1  # (turn of Muzero: 0-first, 1-second)
        self.opponent="expert"  # see if influences anything
        # see if influences anything
        
        ### Self-Play
        self.num_workers=1 # number of simulations self-playing at the same time
        
        self.selfplay_on_gpu=False  #Posso substituir no self_play e verificar se tem mais um lado 
        
        self.max_moves=50  # max moves permited if game is not finished before  (adapt on diferent boards)
        self.num_simulations=25  # number of simulations of future moves
        self.discount=0.8  #  discount factor on future rewards (between 0 and 1) 1- treats future rewards equally, 0-only cares about immediate rewards
        self.temperature_threshold=None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action)
        
        # Root prior exploration noise
        self.root_dirichlet_alpha=0.1  # quantity of random noise, used to encourage exploration during the initial stages of the search
        self.root_exploration_fraction=0.25  # porportion of total simulations for exploration at the root of the tree dedicated to explore actions based on the previous noise
        
        #UCB formula
        self.pb_c_base=19652  # this value is high to prioritize exploration over exploitation, number of times the node have been visited
        self.pb_c_init=1.25  # controls the trade-off between exploration an exploitation,   larger values encourage exploration, smaller-> exploitation of the best known actions
        
        ### Network 
        
        self.network="fullyconnected"  # type of network we are using  (we could also use resnet)
        self.support_size=10   #  ver melhor o que é
        
        # Residual Network  (ver melhor esta parte e se deva coloca la)
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size=32
        self.fc_representation_layers=[]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers=[16]       # Define the hidden layers in the dynamics network
        self.fc_reward_layers=[16]        # Define the hidden layers in the reward network
        self.fc_value_layers=[]           # Define the hidden layers in the value network
        self.fc_policy_layers=[]          # Define the hidden layers in the policy network
        
        ### Training
        self.results_path=pathlib.Path(__file__).resolve().parents[1] / "logs" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") # Path to store the model weights and TensorBoard logs
        self.save_model=True                           # save the model_checkpoint to later continue the training from that point
        self.training_steps=3000000                      # total number of training steps  
        self.batch_size=64                             # number of parts of the game to train at each training step
        self.checkpoint_interval=10                    # number of training steps before using the model for self-playing
        self.value_loss_weight=0.25                    # used to scale the value loss during the training
        self.train_on_gpu=torch.cuda.is_available()    # try to use GPU      verificar se vale a pena
        
        self.optimizer="Adam"  # we also could use SGD
        self.weight_decay=1e-4 # L2 weights regularization
        #self.momentum=0.9 # used only if optimizer is SGD
        
        # Exponential learning rate schedule
        self.lr_init=0.003 # Initial learning rate
        self.lr_decay_rate=1  # set it to 1 to use a constant learning rate
        self.lr_decay_steps=1000  # number of training steps after which the learning rate will be decayed (the learning rate is adjusted according to the decay strategy)
        
        ### Replay Buffer
        
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay, select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case (all experiences have equal probability of being sampled),
        # and 1 corresponds to full prioritization based on the Temporal Difference error magnitudes
        
        # Reanalyze 
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (ver melhor isto)
        self.reanalyse_on_gpu = False    # ver se é util
        
        
        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        
        
    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Go()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        #self.env.render()
        #input("Press enter to take a step ")
        self.env.render()
        #input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                print(self.legal_actions())
                #print(self.board)
                row = int(
                    input(
                        f"Enter the x [0,7] to play for the player {self.env.turn}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the y [0,7] to play for the player {self.env.turn}: "
                    )
                )
                #choice = (row - 1) * 3 + (col - 1)
                choice = row*tam+col
                if (
                    choice in self.legal_actions()
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        for x in range(tam):
            for y in range(tam):
                if x * tam + y == action_number:
                    row = x
                    col = y
        if action_number == tam*tam:
            row = tam
            col = 0
        return f"Play row {row}, column {col}"
    
    
class Go:
    def __init__(self):
        self.possible=[]
        self.passe = 0
        self.board = np.zeros(shape=(tam, tam))
        self.last = np.zeros(shape=(tam, tam))
        self.game_is_over=False 
        self.turn = 1
        self.pp=5
        self.score_black = 0
        self.score_white = 0
        self.Black=41
        self.White=40 # Para mapa 7x7
        
        
    def to_play(self):
        return 2 if self.turn == 1 else 1
    
    def legal_actions(self):
        legal = []

        for i in range(tam):
            for j in range(tam):
                self.visited = [(i,j)]
                board_new = np.copy(self.board)
                if self.confirm_jog(board_new, (i,j)):
                    legal.append(i*tam+j)
        for i in range(self.pp):
            legal.append(tam*tam) #jogada de passar o turno
        return legal
    
    def reset(self):
        #print(self.board) 
        self.possible=[]
        self.passe = 0
        self.board = np.zeros(shape=(tam, tam))
        self.last = np.zeros(shape=(tam, tam))
        self.game_is_over=False 
        self.turn = 1
        self.pp=5
        self.score_black = 0
        self.score_white = 0
        self.Black=25
        self.White=24 # Para mapa 7x7
        
        return self.get_observation()
    
    def step(self, a):
        if a == tam*tam:
            self.passe += 1
            self.pp -= 1
            self.turn = self.change()
            done = self.game_over()
            reward = self.score_white - self.score_black if done else -1

        else:
            for x in range(tam):
                for y in range(tam):
                    if x * tam + y == a:
                        action = (x,y)

            if self.pos_clear(action):
                t=self.jogada(action)
                if(t):
                    self.passe = 0 
                    done = self.game_over()
                    reward = self.score_white - self.score_black if done else 0

        return self.get_observation(), reward, done
    
    def get_observation(self):
        # Obtain the dimensions of the list
        rows = tam
        columns = tam
        # Create a 3D list with shape (3, rows, columns) and initialize with values from self.BoardMatrix
        observation = [[[0 for _ in range(columns)] for _ in range(rows)] for _ in range(3)]

        # Encode player 1 pieces as 1 in the first channel
        for i in range(rows):
            for j in range(columns):
                observation[0][i][j] = 1 if self.board[i][j] == 1 else 0

        # Encode player 2 pieces as 1 in the second channel
        for i in range(rows):
            for j in range(columns):
                observation[1][i][j] = 1 if self.board[i][j] == 2 else 0

        # Encode the current player's pieces as 1 in the third channel
        for i in range(rows):
            for j in range(columns):
                observation[2][i][j] = self.turn
        return observation
     
    def change(self):
        if self.turn  == 1:
            return 2
        else:
            return 1
        
    def set_peca(self, coor):
        if self.turn==1:
            self.board[coor[0]][coor[1]] = 1
        else:
            self.board[coor[0]][coor[1]] = 2
    
    def jogada(self, coor):
        self.visited = [coor]

        self.set_peca(coor)
        if not self.check_ko(coor): 
            
            if self.legal_move(coor) and self.check_pecas():
                self.turn = self.change()
                self.check_cap(coor)    #como a turn ja foi mudada envio so a turn pq quero ver quantas peças adversarias ficam rodeadas
                
                
            elif self.check_cap(coor):  
                self.turn = self.change()
            #else:
            #    self.board[coor[0]][coor[1]]=0
                #print("Jogada inválida")
            #    return False

            if self.turn == 2:
                self.Black -= 1
            else:
                self.White -= 1
                
            if(self.Black==0 and self.turn==1):
                #print("ficou sem peças o jogador Preto")
                #turn=self.change(turn)
                flag1=True

            elif(self.White==0 and self.turn==2):
                #print("ficou sem peças o jogador Branco")
                #turn=self.change(turn)
                flag2= True
            #if(flag1 and flag2):
            #self.game_over()
            return True 
        return False
            # A mensagem na interface e outras operações após uma jogada válida

    def confirm_jog(self,board_new, coor):            
        #Verificar se local é 0
        if board_new[coor[0]][coor[1]]!=0:
            return False
        
        #Efetuar jogada na copia do tabuleiro
        if self.turn==1:
            board_new[coor[0]][coor[1]] = 1
        else:
            board_new[coor[0]][coor[1]] = 2

        #Verificar check_ko
        if not np.all(self.last == np.zeros(shape=(tam, tam))):
            if np.all(self.last == board_new):
                return False

        #Verificar legal_move
        if not self.legal_move2(coor,board_new):
            return False
        
        #Verificar check_pecas 
        if self.turn==1:
            if not self.Black>0:
                return False
        if self.turn==2:
            if not self.White>0:
                return False 
        
        return True  

    def check_ko(self,coor):
        if np.all(self.last == np.zeros(shape=(tam, tam))):
            self.last = copy.deepcopy(self.board)
            return False
        if np.all(self.last == self.board):
            #print("Repetição de estado (ko). Jogada inválida.")
            self.board[coor[0]][coor[1]]=0
            return True
        self.last = copy.deepcopy(self.board)
        return False
    
    def game_over(self):
        if(self.passe>=2 or (self.Black==0 and self.White==0)):
            self.score()
            self.game_is_over=True
            return True
        return False
    
    def score(self):
        self.territorio()
        self.score_white=5.5+self.score_white
        self.score_black=0+self.score_black
        #print(self.score_black)
        #print(self.score_white)
        '''
        if(self.score_white>self.score_black):
            print("Brancas Ganham")  
            #print(self.board) 
        elif(self.score_white<self.score_black):
            print("Pretas Ganham")
        else:
            print("Empate")
        '''
    
    
        ####-------------------------------------------- Testar se funciona------------------------------------------####
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
            
    def check_pecas(self):
        if(self.turn==1 and self.Black>0):
            return True
        if(self.turn==2 and self.White>0):
            return True
        return False
            

    #def capture(self):
    #    if(self.turn==2):
    #        self.score_black+=len(self.cap)
    #    else:
    #        self.score_white+=len(self.cap)
    #    for stone in self.cap:
    #        self.board[stone[0]][stone[1]]=0
            
      
    def check_cap(self,coor):  
        
        _,neigh=self.liberties(coor)
        for stone in neigh:
            self.cap=[]
            if (self.board[stone[0]][stone[1]]== self.turn):
                if(self.capture(stone)):
                    #print(self.cap)
                    for pedra in self.cap:
                        self.board[pedra[0]][pedra[1]]=0
                        if self.turn ==1:
                            self.score_white+=1
                        else:
                            self.score_black+=1
                    return True

        return False
                
            
    def capture(self,stone): 
        stack=list()
        stack.append(stone)
        #print(stack)   # adicionar o primeiro elemento
        while(stack):
            stone=stack.pop()
            
            if(self.board[stone[0]][stone[1]]==self.turn and not any(np.array_equal(stone, item) for item in self.cap)):
                val,neigh=self.liberties(stone)
                if val:
                    self.cap=[]
                    return False
                self.cap.append(stone)
                stack.extend(neigh)       # adicionar os vizinhos todos à lista
                #print(neigh)
        return True
    
    
    
    def legal_move(self,coor):
        
        val,neigh=self.liberties(coor)
        if val:
            return True

        self.set_peca(coor)  #como chamo a recursivamente esta funçao no check_color tenho que  ter isto aqui
        
        if(self.check_color(neigh)):
            return True
        
        
        return False
    
    def legal_move2(self,coor,board_n):
            
        val,neigh=self.liberties_legal(coor,board_n)
        if val:
            return True

        if self.turn==1:
            board_n[coor[0]][coor[1]] = 1
        else:
            board_n[coor[0]][coor[1]] = 2
        
        if(self.check_color2(neigh, board_n)):
            return True
        
        
        return False
    
    def check_color(self,neigh):
        
        for stone in neigh:
            
            
            if (self.board[stone[0]][stone[1]]== self.turn and not any(np.array_equal(stone, item) for item in self.visited)):  # é criado uma variavel item que percorre o self.visited e ve se exsite algum valor igual à
                #print("entrou" +str(stone))
                #print(self.board[stone[0]][stone[1]])
                #print(self.board)
                self.visited.append(stone)
                if (self.legal_move(stone)):
                    return True
                  
        return False
    
    def check_color2(self,neigh, board_n):
        
        for stone in neigh:
            
            
            if (board_n[stone[0]][stone[1]]== self.turn and not any(np.array_equal(stone, item) for item in self.visited)):  # é criado uma variavel item que percorre o self.visited e ve se exsite algum valor igual à
                #print("entrou" +str(stone))
                #print(self.board[stone[0]][stone[1]])
                #print(self.board)
                self.visited.append(stone)
                if (self.legal_move2(stone,board_n)):
                    return True
                  
        return False
              

    def liberties(self,coor): # verifica se existem liberdades e os vizinhos da mesma cor
        neigh=[]
        c=0
        for element in [EAST,SOUTH,WEST,NORTH]:
            pos=(coor[0]+element[0],coor[1]+element[1])
            if(0<=pos[0]<tam and 0<=pos[1]<tam):
                #neigh.append(pos)
                if(self.board[pos[0]][pos[1]]==0):
                    c+=1
                elif(self.board[pos[0]][pos[1]]==1 and self.turn==1):
                    #print("entrou 1")
                    neigh.append(pos)
                elif(self.board[pos[0]][pos[1]]==2 and self.turn==2):
                    #print("entrou 2")
                    neigh.append(pos)
                    

        if c>0:
            return True,neigh
        
        return False,neigh

    def liberties_legal(self,coor, board_n): # verifica se existem liberdades e os vizinhos da mesma cor
        neigh=[]
        c=0
        for element in [EAST,SOUTH,WEST,NORTH]:
            pos=(coor[0]+element[0],coor[1]+element[1])
            if(0<=pos[0]<tam and 0<=pos[1]<tam):
                #neigh.append(pos)
                if(board_n[pos[0]][pos[1]]==0):
                    c+=1
                elif(board_n[pos[0]][pos[1]]==1 and self.turn==1):
                    #print("entrou 1")
                    neigh.append(pos)
                elif(board_n[pos[0]][pos[1]]==2 and self.turn==2):
                    #print("entrou 2")
                    neigh.append(pos)
                    

        if c>0:
            return True,neigh
        
        return False,neigh
    
    def pos_clear(self, coor):
        return self.board[coor[0]][coor[1]] == 0

    
    def expert_action(self):
        action = np.random.choice(self.legal_actions())
        return action
    
    def render(self):
        print(self.board)
        if self.game_is_over:
            print(f'Black Score: {self.score_black}')
            print(f'White Score: {self.score_white}')
        #print(self.board[::-1])

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
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
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")