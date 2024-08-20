import datetime
import pathlib
import numpy
import torch
from .abstract_game import AbstractGame




# change values to find the best parameters
class MuZeroConfig:
    def __init__(self):
        self.seed=0
        self.max_num_gpus=None
        
        ### Game
        self.observation_shape=(3,5,5) # Dimensions (number of diferent values (player1,player2, empty), height,width)
        self.action_space=list(range(25)) #all possible actions 
        self.players=list(range(2))  # list of players
        self.stacked_observations=0  # Number of previous observations and previous actions to add to the current observation
        
        # evalute
        self.muzero_player=1  # (turn of Muzero: 0-first, 1-second)
        self.opponent="expert"  # see if influences anything
        # see if influences anything
        
        ### Self-Play
        self.num_workers=1 # number of simulations self-playing at the same time
        
        self.selfplay_on_gpu=False  #Posso substituir no self_play e verificar se tem mais um lado 
        
        self.max_moves=30  # max moves permited if game is not finished before  (adapt on diferent boards)
        self.num_simulations=25  # number of simulations of future moves
        self.discount=0.9  #  discount factor on future rewards (between 0 and 1) 1- treats future rewards equally, 0-only cares about immediate rewards
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
        self.training_steps=5000000                      # total number of training steps  
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
        self.env = Ataxx()

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

    
    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()
    
    
class Ataxx:
    def __init__(self):
        self.row=5
        self.column=5
        self.possible=[]
        self.board = [[] for i in range(self.row)]
        for row in range(self.row):
            for collumn in range(self.column):
                self.board[row].append(0)
        self.board[0][0] = -1
        self.board[4][0] = 1
        self.board[0][4] = 1
        self.board[4][4] = -1
        
        self.player = 1
    
    # check which turn is to use in MCTS    
    def to_play(self):
        return 0 if self.player == 1 else 1
    
    # restart the board when a game finish during training
    def reset(self):
        self.possible=[]
        self.board = [[] for i in range(self.row)]
        for row in range(self.row):
            for collumn in range(self.column):
                self.board[row].append(0)
        self.board[0][0] = -1
        self.board[4][0] = 1
        self.board[0][4] = 1
        self.board[4][4] = -1
        self.player = 1
        return self.get_observation()
    
    # execute Muzero action
    def step(self, action):
            
            
            flag=True
            t=self.legal_actions()
            i=0
                
            while i<=len(t)-1 and flag:
                    
                if(t[i]==action):
                    index=i
                    flag=False
                i+=1
                
            action=self.possible[index]
            start_position, end_position = action
            piecevalue = self.PieceAt(end_position)
            distanceX, distanceY = self.GetDistanceBoardUnits(start_position, end_position)
            if (distanceX == 1 and distanceY <= 1) or (distanceY == 1 and distanceX <= 1):
                    if piecevalue == 0:
                        self.MakeNewPieceAt(end_position, -1)
                        self.CatchPiece(end_position, -1)
                            
            elif (distanceX == 2 and distanceY <= 2) or (distanceY == 2 and distanceX <= 2):
                    if piecevalue == 0:
                        self.MovePieceTo(start_position, end_position, -1)
                        self.CatchPiece(end_position, -1)
    
            
            done = self.have_winner() or len(self.legal_actions()) == 0

            if(self.have_winner()):
                reward=20
            else:
                reward=-2
            
            
            
            self.player *= -1

            return self.get_observation(), reward, done
    
    
    #Given a window position converts it to board position          
    def PieceAt(self, Position):
        row, col = Position
        piece = self.board[row][col]
        return piece
    
    #Calculates the distance between two points
    def GetDistanceBoardUnits(self, StartPosition, EndPosition):
        x1, y1 = StartPosition
        x2, y2 = EndPosition
        distanceX = abs(x2 - x1)
        distanceY = abs(y2 - y1)
        return distanceX, distanceY        

    #Moves piece to the target location
    def MovePieceTo(self, CurrentPosition, TargetPosition, PieceIndex):
        x1, y1 = CurrentPosition
        x2, y2 = TargetPosition
        self.board[x1][y1] = 0
        self.board[x2][y2] = PieceIndex
        pass
    
    #Creates a new piece in the target position
    def MakeNewPieceAt(self, TargetPosition, PieceIndex):
        x, y = TargetPosition
        self.board[x][y] = PieceIndex
        pass
    
    #Capture a piece
    def CatchPiece(self, Position, PlayerIndex):
        x, y = Position
        RemovablePositions = []
        TangentPositions = [(x+1, y), (x+1, y+1), (x, y+1),
                            (x-1, y+1), (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)]
        for p in range(len(TangentPositions)):
            xp, yp = TangentPositions[p]
            if not (xp >= 0 and xp < self.row and yp >= 0 and yp < self.column):
                RemovablePositions.append(TangentPositions[p])
                continue
            if self.board[xp][yp] == 0 or self.board[xp][yp] == PlayerIndex:
                RemovablePositions.append(TangentPositions[p])
                continue
        for i in range(len(RemovablePositions)):
            TangentPositions.remove(RemovablePositions[i])
        for a in range(len(TangentPositions)):
            x, y = TangentPositions[a]
            self.board[x][y] = PlayerIndex
    
    
    
    # get the  observations for Muzero
    def get_observation(self):
        # Create a 3D list with shape (3, rows, columns) and initialize with values from self.BoardMatrix
        observation = [[[0 for _ in range(self.column)] for _ in range(self.row)] for _ in range(3)]

        # Encode player 1 pieces as 1 in the first channel
        # Encode player 2 pieces as 1 in the second channel
        # Encode the current player's pieces in the third channel
        for i in range(self.row):
            for j in range(self.column):
                observation[0][i][j] = 1 if self.board[i][j] == 1 else 0
                observation[1][i][j] = 1 if self.board[i][j] == -1 else 0
                observation[2][i][j] = self.player

        return observation
     
    # check all possible moves for AI
    def legal_actions(self):
        self.possible=[]
        legal = []
        #print(self.player)
        for row in range(self.row):
            for col in range(self.column):
                #print("oi")
                #print(self.board[row][col])
                if self.board[row][col] == self.player:
                    # Add legal actions for each piece of the current player
                    legal.extend(self.get_legal_moves_for_piece((row, col)))
            #print("/n")        
        return legal

    # check possible moves for each piece
    def get_legal_moves_for_piece(self, position):
        initial_position = position
        legal_moves = []
        for col in range(max(0, position[0] - 2), min(self.column, position[0] + 3)):
            for row in range(max(0, position[1] - 2), min(self.row, position[1] + 3)):
                if self.board[col][row] == 0:
                    self.possible.append((initial_position, (col, row)))
                    legal_moves.append(col*self.column+row)
        return legal_moves
    
    
    # Get the piece tangent positions given a Radius and exclude out of bounds positions and piece occupied positions
    def GetTangentPositionsOfPiece(self, Position, Radius: int):
        centerX, centerY = Position
        Positions = []
        # GetVerticalPositions with TargetX being CenterX + Radius and CenterX - Radius
        for VP in range(3 + (2*Radius)):
            RightPosition = [centerX + (Radius+1), -(Radius+1) + VP + centerY]
            # check if position is out of bounds
            if(RightPosition[0] >= 0 and RightPosition[0] < len(self.board) and RightPosition[1] >= 0 and RightPosition[1] < len(self.board[0])):
                # check if position is available
                if(self.board[RightPosition[0]][RightPosition[1]] == 0):
                    Positions.append(RightPosition)
            LeftPosition = [centerX - (Radius+1), -(Radius+1) + VP + centerY]
            if(LeftPosition[0] >= 0 and LeftPosition[0] < len(self.board) and LeftPosition[1] >= 0 and LeftPosition[1] < len(self.board[0])):
                # check if position is available
                if(self.board[LeftPosition[0]][LeftPosition[1]] == 0):
                    Positions.append(LeftPosition)

        # GetHorizontalPositions with TargetY being CenterY + Radius and CenterY - Radius
        for HP in range(1 + (2*Radius)):
            TopPosition = [-Radius + HP + centerX, centerY - (Radius+1)]
            # check if position is out of bounds
            if(TopPosition[0] >= 0 and TopPosition[0] < len(self.board) and TopPosition[1] >= 0 and TopPosition[1] < len(self.board[0])):
                # check if position is available
                if(self.board[TopPosition[0]][TopPosition[1]] == 0):
                    Positions.append(TopPosition)
            BottomPosition = [-Radius + HP + centerX, centerY + (Radius+1)]
            # check if position is out of bounds
            if(BottomPosition[0] >= 0 and BottomPosition[0] < len(self.board) and BottomPosition[1] >= 0 and BottomPosition[1] < len(self.board[0])):
                # check if position is available
                if(self.board[BottomPosition[0]][BottomPosition[1]] == 0):
                    Positions.append(BottomPosition)
        return Positions
    
    
    #Check if the game is over
    def have_winner(self):
        isThereTangentPositionsRed = False
        isThereTangentPositionsBlue = False

        isThereMovementPositionsRed = False
        isThereMovementPositionsBlue = False
        for row in range(self.row):
            for col in range(self.column):
                if self.board[row][col] == -1:
                    if len(self.GetTangentPositionsOfPiece((row, col), 1)) > 0:
                        isThereMovementPositionsRed = True
                    if len(self.GetTangentPositionsOfPiece((row, col), 0)) > 0:
                        isThereTangentPositionsRed = True

                elif self.board[row][col] == 1:
                    if len(self.GetTangentPositionsOfPiece((row, col), 1)) > 0:
                        isThereMovementPositionsBlue = True
                    if len(self.GetTangentPositionsOfPiece((row, col), 0)) > 0:
                        isThereTangentPositionsBlue = True
            if (isThereTangentPositionsRed or isThereMovementPositionsRed) and (isThereMovementPositionsBlue or isThereTangentPositionsBlue):
                break
       
        return (isThereTangentPositionsRed + isThereMovementPositionsRed) * (isThereMovementPositionsBlue + isThereTangentPositionsBlue)
    

    def expert_action(self):
        action = numpy.random.choice(self.legal_actions())
        return action
    
