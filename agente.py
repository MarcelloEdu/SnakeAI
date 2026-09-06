import torch
import random
import numpy as np
import os
from collections import deque
from snakeAI import SnakeGameAI, Direction, Point, BLOCK_SIZE
from modelo import Linear_QNet, QTrainer


MAX_MEMORY = 100_000 # Quantidade de jogadas que a IA lembra
BATCH_SIZE = 1000    # Tamanho do lote de treinamento
LR = 0.001           # Taxa de aprendizado (Learning Rate)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Controle de aleatoriedade (Exploration vs Exploitation)
        self.gamma = 0.9 # Taxa de desconto (Peso das recompensas futuras)
        #deque pois precisamos de uma estrutura de dados de fila circular, 
        #cada vez que a cobra da um passo nos salvamos a tupla (state, action, reward, next_state, done)
        #na memória. Se a memória estiver cheia, a tupla mais antiga é descartada.
        self.memory = deque(maxlen=MAX_MEMORY) # Se encher, apaga as memórias mais antigas
        
        self.model = Linear_QNet(11, 256, 3)
        caminho_modelo = 'model/model.pth'
        if os.path.exists(caminho_modelo):
            print("Carregando modelo existente...")
            self.model.load_state_dict(torch.load(caminho_modelo))
            self.model.eval() # Trava a rede em "Modo de Avaliação" (desativa atualizações)
            print("Modelo carregado com sucesso!")
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """Traduz o jogo em 11 sensores booleanos (0 ou 1)"""
        head = game.snake[0]
        
        # Pontos ao redor da cabeça
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Direção atual
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 1. Perigo à frente?
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # 2. Perigo à direita?
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # 3. Perigo à esquerda?
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # 4. Direção atual (4 sensores, apenas 1 será True)
            dir_l, dir_r, dir_u, dir_d,
            
            # 5. Onde está a comida em relação à cabeça?
            game.food.x < game.head.x,  # Comida à esquerda
            game.food.x > game.head.x,  # Comida à direita
            game.food.y < game.head.y,  # Comida acima
            game.food.y > game.head.y   # Comida abaixo
        ]

        # Converte booleanos [True, False] para inteiros [1, 0]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Guarda a jogada na memória para treinar depois."""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """Decide o que fazer com base no estado atual."""
        self.epsilon = 80 - self.n_games # Quanto mais jogos, menor o epsilon
        final_move = [0, 0, 0]
        
        # Exploration: Faz um movimento aleatório
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        
        # Exploitation: Usa a inteligência da rede neural
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # Prevê a melhor ação (ativaremos isso depois)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

    def train_long_memory(self):
        """Treina a rede com um lote de memórias antigas (Experience Replay)"""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        # A função zip(*...) agrupa todos os estados, ações, etc., em tuplas separadas
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)