import pathlib
import torch
import datetime

# change values to find the best parameters
class MuZeroConfig:
    def __init__(self):
        self.seed=0
        self.max_num_gpus=None
        
        ### Game
        self.observation_shape=(3,6,6) # Dimensions (number of diferent values (player1,player2, empty), height,width)
        self.action_space=list(range(36)) #all possible actions 
        self.players=list(range(2))  # list of players
        self.stacked_observations=0  # Number of previous observations and previous actions to add to the current observation
        
        # evalute
        self.muzero_player=1  # (turn of Muzero: 0-first, 1-second)
        self.opponent="expert"  # see if influences anything
        # see if influences anything
        
        ### Self-Play
        self.num_workers=1 # number of simulations self-playing at the same time
        
        self.selfplay_on_gpu=False  #Posso substituir no self_play e verificar se tem mais um lado 
        
        self.max_moves=42  # max moves permited if game is not finished before  (adapt on diferent boards)
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
        self.channels = 32  # Number of channels in the ResNet
        self.reduced_channels_reward = 32  # Number of channels in reward head
        self.reduced_channels_value = 48  # Number of channels in value head
        self.reduced_channels_policy = 48  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network
        
        # # Fully Connected Network
        # self.encoding_size=48
        # self.fc_representation_layers=[]  # Define the hidden layers in the representation network
        # self.fc_dynamics_layers=[68]       # Define the hidden layers in the dynamics network
        # self.fc_reward_layers=[48]        # Define the hidden layers in the reward network
        # self.fc_value_layers=[]           # Define the hidden layers in the value network
        # self.fc_policy_layers=[]          # Define the hidden layers in the policy network
        
        self.encoding_size = 32  # Ajuste para corresponder à mudança na primeira camada
        self.fc_representation_layers = []  # Não há camadas específicas definidas, deixe como está ou ajuste conforme necessário
        self.fc_dynamics_layers = [16]  # Ajuste para corresponder às dimensões corretas da dynamics_encoded_state_network
        self.fc_reward_layers = [16]  # Ajuste para corresponder às dimensões corretas da dynamics_reward_network
        self.fc_value_layers = []  # Ajuste para corresponder às dimensões corretas da prediction_value_network
        self.fc_policy_layers = []  # Ajuste para corresponder às dimensões corretas da prediction_policy_network

        
        ### Training
        self.results_path=pathlib.Path(__file__).resolve().parents[1] / "logs" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") # Path to store the model weights and TensorBoard logs
        self.save_model=True                           # save the model_checkpoint to later continue the training from that point
        self.training_steps=500000                      # total number of training steps  
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