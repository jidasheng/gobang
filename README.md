A Gobang(also known as "Five in a Row" and "Gomoku") game equipped with AlphaGo-liked AI.

# Features
- Friendly GUI
- MATCH RULES supported
- Easy to install
- AlphaGo-liked AI

# Game
![Screen](https://user-images.githubusercontent.com/44913901/71503260-6e1cf200-28af-11ea-9e59-ef6b9e86d5fa.png)
### AI Players
- `Greedy AI`: the POLICY network without randomness
- `Probabilistic AI`: the POLICY network with randomness
- `Thinking AI(xxx s)`: MCTS algorithm with POLICY and VALUE networks
    - To be updated

# Install
- dependencies
    - Python 3
    - [PyTorch][1]
- install
    ```sh
    $ pip install gobang
    ```
# Run
```sh
$ gobang

# or
$ python -m gobang
```

# Comments
- This project was originally developed in 2017 based on [RocAlphaGo][2].
- The AI is developed based on Supervised Learning and Reinforcement learning from [AlphaGo][3].

# References
1. [RocAlphaGo][2]
2. [Mastering the game of Go with deep neural networks and tree search][3]

[1]:https://pytorch.org/
[2]:https://github.com/Rochester-NRT/RocAlphaGo
[3]:https://www.nature.com/articles/nature16961