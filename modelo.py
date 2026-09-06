import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Camada de entrada (11 sensores) -> Camada oculta
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Camada oculta -> Camada de saída (3 ações)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # A função de ativação ReLU zera valores negativos e mantém os positivos
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """Salva os pesos (conhecimento) da rede neural em um arquivo."""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Adam é o otimizador que vai ajustar os pesos da rede para reduzir os erros
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Mean Squared Error (Erro Quadrático Médio) mede a diferença entre a predição e o resultado real
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Converte as listas do Python em Tensores do PyTorch
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # Garante que os tensores tenham o formato correto mesmo se passarmos apenas 1 jogada
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Qual era a predição da rede neural para o estado atual?
        pred = self.model(state)

        # 2. Qual deveria ser a predição ideal? (Equação de Bellman)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Q_new = recompensa imediata + (gamma * maior recompensa futura esperada)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Atualiza o valor alvo apenas da ação que foi tomada
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 3. Calcula o erro e ajusta os pesos da rede (Backpropagation)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()