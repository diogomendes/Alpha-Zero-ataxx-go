import pygame
import random as rand
import math
import numpy as np
import models
import torch
import pathlib




#-------------------------- variables used for the interface------------------------------------------------



SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY=(200,200,200)

pygame.font.init()
font = pygame.font.SysFont("Arial", 48)
font.set_bold(True)


window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(("ATTAXX"))

# PROJECT DESCRIPTION
# for the board in the project, RED pieces are defines as the number -1 and BLUE pieces are defined as the number 1 in the board
# the board is represented by a matrix composed of three numbers which corespond to existing a piece in that position or not(-1 or 1 for existing and 0 for not existing)

#--------------------------------------- Importante functions for the game --------------------

class Board:  
    
    def __init__(self, Row_count: int, Collumn_Count: int):
        self.square_edge_size = SCREEN_WIDTH//Collumn_Count
        self.square_size = self.square_edge_size ** 2
        self.row_count = Row_count
        self.collumn_count = Collumn_Count
        self.BoardMatrix = [[] for i in range(self.row_count)]
        self.player=-1
        self.vict=0
        self.CreateBoard()

#---------------------------------- create board and draw everything on the interface --------------------------


    def CreateBoard(self):
        for row in range(self.row_count):
            for collumn in range(self.collumn_count):
                self.BoardMatrix[row].append(0)
        self.BoardMatrix[0][0] = -1
        self.BoardMatrix[self.collumn_count-1][0] = 1
        self.BoardMatrix[0][self.row_count-1] = 1
        self.BoardMatrix[self.collumn_count-1][self.row_count-1] = -1
        
    
        
    
    def DrawTable(self):
        window.fill(BLACK)
        for row in range(self.row_count):
            for col in range(self.collumn_count):
                pygame.draw.rect(window, "TAN", (row*self.square_edge_size, col *
                                self.square_edge_size, self.square_edge_size*0.98, self.square_edge_size*0.98))
                

    def DrawPieces(self):
        #print("2")
        #print(self.BoardMatrix)
        for col in range(self.collumn_count):
            for row in range(self.row_count):
                if(self.BoardMatrix[col][row] == -1):
                    pygame.draw.circle(
                        window, RED, (self.square_edge_size//2 + (2 * row * self.square_edge_size//2), self.square_edge_size//2 + (2 * col * self.square_edge_size//2)),  self.square_edge_size//4)
                    pygame.draw.circle(
                        window, BLACK, (self.square_edge_size//2 + (2 * row * self.square_edge_size//2), self.square_edge_size//2 + (2 * col * self.square_edge_size//2)),  self.square_edge_size//4,width=2)
                elif(self.BoardMatrix[col][row] == 1):
                    pygame.draw.circle(
                        window, BLUE, (self.square_edge_size//2 + (2 * row * self.square_edge_size//2), self.square_edge_size//2 + (2 * col * self.square_edge_size//2)), self.square_edge_size//4)
                    pygame.draw.circle(
                        window, BLACK, (self.square_edge_size//2 + (2 * row * self.square_edge_size//2), self.square_edge_size//2 + (2 * col * self.square_edge_size//2)),  self.square_edge_size//4,width=2)
                elif(self.BoardMatrix[col][row]==8):
                    pygame.draw.circle(
                        window, GRAY, (self.square_edge_size//2 + (2 * row * self.square_edge_size//2), self.square_edge_size//2 + (2 * col * self.square_edge_size//2)),  self.square_edge_size//4)
                    pygame.draw.circle(
                        window, BLACK, (self.square_edge_size//2 + (2 * row * self.square_edge_size//2), self.square_edge_size//2 + (2 * col * self.square_edge_size//2)),  self.square_edge_size//4,width=2)
                                    
    
    #------------------------------- functions to check end game and the winner --------------------------------------
    
    # Print the result of the match               
    def end(self):
        self.vict=self.winner()
        if(self.vict!=0):
            end_game_text = font.render(f"Winner player {self.vict}", True, WHITE)
        else:
            end_game_text = font.render("Draw", True, WHITE)
        text_rect = end_game_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        window.blit(end_game_text, text_rect)  
       
     
    # check who wins or if is a draw 
    def winner(self):
        jog1=0
        jog2=0
        for i in range(self.row_count):
            for j in range(self.collumn_count):  
                    if(self.BoardMatrix[i][j]==1):
                        jog1+=1
                    elif(self.BoardMatrix[i][j]==-1):
                        jog2+=1
        if(jog2<jog1):
            return 1
        elif(jog2>jog1):
            return 2
        return 0  
    
    #Checks if the game is over
    def CheckGame(self):
        isThereTangentPositionsRed = False
        isThereTangentPositionsBlue = False

        isThereMovementPositionsRed = False
        isThereMovementPositionsBlue = False
        for row in range(self.row_count):
            for col in range(self.collumn_count):
                if self.BoardMatrix[row][col] == -1:
                    if len(self.GetTangentPositionsOfPiece((row, col), 1)) > 0:
                        isThereMovementPositionsRed = True
                    if len(self.GetTangentPositionsOfPiece((row, col), 0)) > 0:
                        isThereTangentPositionsRed = True

                elif self.BoardMatrix[row][col] == 1:
                    if len(self.GetTangentPositionsOfPiece((row, col), 1)) > 0:
                        isThereMovementPositionsBlue = True
                    if len(self.GetTangentPositionsOfPiece((row, col), 0)) > 0:
                        isThereTangentPositionsBlue = True
            if (isThereTangentPositionsRed or isThereMovementPositionsRed) and (isThereMovementPositionsBlue or isThereTangentPositionsBlue):
                break
        
        return (isThereTangentPositionsRed + isThereMovementPositionsRed) * (isThereMovementPositionsBlue + isThereTangentPositionsBlue)       
                    

#--------------------------  execute moves of the players ------------------------------
                    
    #given a window position converts it to board position  
    def ConvertToBoardValues(self, Position):
        x, y = Position
        row = y // self.square_edge_size
        col = x // self.square_edge_size
        return row, col


    #Given a window position converts it to board position
    def PieceAt(self, Position):
        row, col = Position
        piece = self.BoardMatrix[row][col]
        return piece

    #Moves piece to the target location
    def MovePieceTo(self, CurrentPosition, TargetPosition, PieceIndex):
        x1, y1 = CurrentPosition
        x2, y2 = TargetPosition
        self.BoardMatrix[x1][y1] = 0
        self.BoardMatrix[x2][y2] = PieceIndex
        pass

    #Creates a new piece in the target position
    def MakeNewPieceAt(self, TargetPosition, PieceIndex):
        x, y = TargetPosition
        self.BoardMatrix[x][y] = PieceIndex
        pass

    # Get the piece tangent positions given a Radius and exclude out of bounds positions and piece occupied positions
    def GetTangentPositionsOfPiece(self, Position, Radius: int):
        centerX, centerY = Position
        Positions = []
        # GetVerticalPositions with TargetX being CenterX + Radius and CenterX - Radius
        for VP in range(3 + (2*Radius)):
            RightPosition = [centerX + (Radius+1), -(Radius+1) + VP + centerY]
            # check if position is out of bounds
            if(RightPosition[0] >= 0 and RightPosition[0] < len(self.BoardMatrix) and RightPosition[1] >= 0 and RightPosition[1] < len(self.BoardMatrix[0])):
                # check if position is available
                if(self.BoardMatrix[RightPosition[0]][RightPosition[1]] == 0):
                    Positions.append(RightPosition)
            LeftPosition = [centerX - (Radius+1), -(Radius+1) + VP + centerY]
            if(LeftPosition[0] >= 0 and LeftPosition[0] < len(self.BoardMatrix) and LeftPosition[1] >= 0 and LeftPosition[1] < len(self.BoardMatrix[0])):
                # check if position is available
                if(self.BoardMatrix[LeftPosition[0]][LeftPosition[1]] == 0):
                    Positions.append(LeftPosition)

        # GetHorizontalPositions with TargetY being CenterY + Radius and CenterY - Radius
        for HP in range(1 + (2*Radius)):
            TopPosition = [-Radius + HP + centerX, centerY - (Radius+1)]
            # check if position is out of bounds
            if(TopPosition[0] >= 0 and TopPosition[0] < len(self.BoardMatrix) and TopPosition[1] >= 0 and TopPosition[1] < len(self.BoardMatrix[0])):
                # check if position is available
                if(self.BoardMatrix[TopPosition[0]][TopPosition[1]] == 0):
                    Positions.append(TopPosition)
            BottomPosition = [-Radius + HP + centerX, centerY + (Radius+1)]
            # check if position is out of bounds
            if(BottomPosition[0] >= 0 and BottomPosition[0] < len(self.BoardMatrix) and BottomPosition[1] >= 0 and BottomPosition[1] < len(self.BoardMatrix[0])):
                # check if position is available
                if(self.BoardMatrix[BottomPosition[0]][BottomPosition[1]] == 0):
                    Positions.append(BottomPosition)
        return Positions

    #Calculates the distance between two points
    def GetDistanceBoardUnits(self, StartPosition, EndPosition):
        x1, y1 = StartPosition
        x2, y2 = EndPosition
        distanceX = abs(x2 - x1)
        distanceY = abs(y2 - y1)
        return distanceX, distanceY


    def DrawBoard(self):
        self.DrawTable()
        self.DrawPieces()


    #Capture a piece 
    def CatchPiece(self, Position, PlayerIndex):
        x, y = Position
        RemovablePositions = []
        TangentPositions = [(x+1, y), (x+1, y+1), (x, y+1),
                            (x-1, y+1), (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)]
        for p in range(len(TangentPositions)):
            xp, yp = TangentPositions[p]
            if not (xp >= 0 and xp < self.collumn_count and yp >= 0 and yp < self.row_count):
                RemovablePositions.append(TangentPositions[p])
                continue
            if self.BoardMatrix[xp][yp] == 0 or self.BoardMatrix[xp][yp] == PlayerIndex:
                RemovablePositions.append(TangentPositions[p])
                continue
        for i in range(len(RemovablePositions)):
            TangentPositions.remove(RemovablePositions[i])
        for a in range(len(TangentPositions)):
            x, y = TangentPositions[a]
            self.BoardMatrix[x][y] = PlayerIndex

#--------- determine the possible moves for the player and draw them on the interface ---------------------------    
    
    # draw possible moves after selecting on piece
    def possibles(self,position,turn):
        
        if(self.BoardMatrix[position[0]][position[1]]==turn):
            for col in range(max(0,position[0] - 2),min(self.collumn_count,position[0]+ 3)):
                for row in range(max(0,position[1]-2),min(self.row_count, position[1]+3)):
                    if(self.BoardMatrix[col][row]==0):
                        self.BoardMatrix[col][row]=8
            self.DrawPieces()
    
    # Clean possible moves
    def clear_pos(self):
        for col in range(self.collumn_count):
            for row in range(self.row_count):
                if(self.BoardMatrix[col][row] == 8):
                    self.BoardMatrix[col][row]=0
        self.DrawPieces()   
        
        
#--------------------- check possible moves for the AI  --------------------------------------------
            
    # check all possible moves for AI
    def legal_actions(self,ai_player):
        self.possible=[]
        legal = []
        #print(self.player)
        for row in range(self.row_count):
            for col in range(self.collumn_count):
                #print("oi")
                #print(self.board[row][col])
                if self.BoardMatrix[row][col] == ai_player:
                    # Add legal actions for each piece of the current player
                    legal.extend(self.get_legal_moves_for_piece((row, col)))
            #print("/n")        
        return legal

    # check possible moves for each piece
    def get_legal_moves_for_piece(self, position):
        initial_position = position
        legal_moves = []
        for col in range(max(0, position[0] - 2), min(self.row_count, position[0] + 3)):
            for row in range(max(0, position[1] - 2), min(self.collumn_count, position[1] + 3)):
                if self.BoardMatrix[col][row] == 0:
                    self.possible.append((initial_position, (col, row)))
                    legal_moves.append(col*self.collumn_count+row)
        return legal_moves
    

    
    # check which turn is to use in MCTS
    def to_play(self):
        return 0 if self.player == -1 else 1


#-------------------------- important functions for the muzero -------------------------------------------
    
    #Load Muzero
    def load_model(self, checkpoint_path):
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
            
        return self.checkpoint
        


    # get the  observations for Muzero
    def get_observation(self):
        
        # Obtain the dimensions of the list
        rows = len(self.BoardMatrix)
        columns = len(self.BoardMatrix[0])
        
        # Create a 3D list with shape (3, rows, columns) and initialize with values from self.BoardMatrix
        observation = [[[0 for _ in range(columns)] for _ in range(rows)] for _ in range(3)]

        # Encode player 1 pieces as 1 in the first channel
        # Encode player 2 pieces as 1 in the second channel
        # Encode the current player's pieces in the third channel
        for i in range(rows):
            for j in range(columns):
                observation[0][i][j] = 1 if self.BoardMatrix[i][j] == 1 else 0
                observation[1][i][j] = 1 if self.BoardMatrix[i][j] == -1 else 0
                observation[2][i][j] = self.player

        return observation
    
    # execute Muzero action 
    def step(self, action,ai_player):
            flag=True
            t=self.legal_actions(ai_player)
            i=0
            
            while i<=len(t)-1 and flag:
                
                if(t[i]==action):
                    index=i
                    flag=False
                i+=1
           
            action=self.possible[index]  # get the coordinates of the start position and end position 
            start_position, end_position = action
            piecevalue = self.PieceAt(end_position)
            distanceX, distanceY = self.GetDistanceBoardUnits(start_position, end_position)
            if (distanceX == 1 and distanceY <= 1) or (distanceY == 1 and distanceX <= 1):
                    if piecevalue == 0:
                        self.MakeNewPieceAt(end_position, ai_player)
                        self.CatchPiece(end_position, ai_player)
                        
            elif (distanceX == 2 and distanceY <= 2) or (distanceY == 2 and distanceX <= 2):
                    if piecevalue == 0:
                        self.MovePieceTo(start_position, end_position, ai_player)
                        self.CatchPiece(end_position, ai_player)
                        
   
    
                  
#----------------------------- Use mcts and game history to find the best action to Muzero execute -----------------------------------
    
    
class selffplay():
    def __init__(self, initial_checkpoint, seed,board,config):
        self.config = config
        self.board=board
        
        # Fix random generator seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cpu"))
        self.model.eval()
        
        
        
    def play_game(self, temperature, temperature_threshold,ai_player):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.board.get_observation()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        #game_history.to_play_history.append(self.game.to_play())

        done = False

        

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(np.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(np.array(observation).shape)} dimensionnal. Got observation of shape: {np.array(observation).shape}"
                assert (
                    np.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {np.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )
    
                # Choose the action
                
                root, mcts_info = MCTS(self.config).run(
                        self.model,
                        stacked_observations,
                        self.board.legal_actions(ai_player),
                        self.board.to_play(),
                        True,
                )
                action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                )
                done=True
                self.board.step(action,ai_player)
        return game_history
    
    def select_action(self,node, temperature):
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

        return action
    
    
    
# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = np.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            np.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = np.concatenate(
                    (
                        np.zeros_like(self.observation_history[index]),
                        [np.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = np.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    
        
