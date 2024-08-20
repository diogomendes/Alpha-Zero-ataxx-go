# Alpha Zero to Ataxx and Go

## Project Compatibility and Requirements

### Operating Systems:

Compatible with Windows and Linux

### Dependencies:

Python        -- 3.11.9
numpy         -- 1.26.2
ray           -- 2.9.0
torch         -- 2.1.2
pygame        -- 2.3.0
tensorflow    -- 2.12.0
customtkinter -- 5.2.1

## Instructions to Run the Code:

### Installation of Dependencies:

Ensure Python 3.11.9 and the other dependencies are installed on your system.

### Running the Scripts:

Not for the competition

Enter the "jogo" folder.

To initialize the program, execute:

python3 muzero2.py

A menu will appear:

1-Play
2-Train

In the case of "Play":

1-Play
    1-Player vs AI
        1-Ataxx
        2-Go
    2-AI vs AI
        1-Ataxx
        2-Go

Next, choose the game board for Ataxx by typing in the terminal:

Game mode: A4x4, A5x5, A6x6

For Go, select the board in the interface.

In the case of "Train":

0. ataxx_4x4
1. ataxx_5x5
2. ataxx_6x6
3. go_7x7
4. go_9x9

Regardless of the chosen option:

0. Train
1. Load pretrained model
2. Exit

To use the latest model for testing, modify the files Player_vs_AI and AI_vs_AI.
To train from the latest model, first load the latest model and then train the model.

For the competition, use the contents of the "competição" folder and run the client. However, it is not adapted for the Go game, and the server is not fully complete.

### Expected Interfaces:

When selecting Ataxx and choosing the board size, this interface will appear, and regardless of the players' selection, we will have a board like this, this one is the 5x5:

![ataxx](https://github.com/user-attachments/assets/33729dab-3f68-4c9c-8bad-2df4fd63d2d4)

For Go, we will have this; the board size is 9x9:

![Go 1](https://github.com/user-attachments/assets/efd02c0f-67a9-47ff-a6ff-5b6dc78b15b1)
![Go 2](https://github.com/user-attachments/assets/f351b470-9a82-40ab-a64b-c5f009275ea1)

### Expected Results:

All the models were trained with the following timesteps:

Ataxx 4x4-> Timesteps: 9.7M 
Ataxx 5x5-> Timesteps: 10M 
Ataxx 6x6-> Timesteps: 9.5M 
Go 7x7-> Timesteps: 1.1M 
GO 9x9-> Timesteps: 3M 

Then, they were tested against a player and a model that chose their actions randomly in 25 games. The results are:

#### Against Human:

Ataxx 4x4-> Vitórias: 20% 
Ataxx 5x5-> Vitórias: 20% 
Ataxx 6x6-> Vitórias: 20% 
Go 7x7-> Vitórias: 50% 
GO 9x9-> Vitórias: 70%


#### Against Random model:

Ataxx 4x4-> Vitórias: 45% 
Ataxx 5x5-> Vitórias: 65% 
Ataxx 6x6-> Vitórias: 50% 
Go 7x7-> Vitórias: 60% 
GO 9x9-> Vitórias: 100%














