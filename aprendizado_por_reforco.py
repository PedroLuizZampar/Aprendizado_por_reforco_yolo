import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Rede Neural usada tanto para Classificação quanto para DQN
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )   
    def forward(self, x):
        return self.fc(x)
# Carrega o modelo de classificação treinado
classificador = NeuralNet(input_dim=10, output_dim=2)
classificador.load_state_dict(torch.load("modelo_classificacao.pth"))
classificador.eval()
# Inicializa redes DQN
policy_net, target_net = NeuralNet(10, 2), NeuralNet(10, 2)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()
# Hiperparâmetros
gamma, epsilon, num_episodes = 0.99, 0.1, 100
def update_model(state, action, reward, next_state, done):
    q_value = policy_net(torch.FloatTensor(state)).gather(1, torch.LongTensor([[action]]))
    with torch.no_grad():
        max_next_q = target_net(torch.FloatTensor(next_state)).max(1)[0]
        target = reward + gamma * max_next_q * (1 - done)
    optimizer.zero_grad()
    loss = criterion(q_value, target.unsqueeze(1))
    loss.backward()
    optimizer.step()
def select_action(state, epsilon):
    return np.random.randint(0, 2) if np.random.rand() < epsilon else policy_net(torch.FloatTensor(state)).argmax().item()
# Loop de treinamento
for episode in range(num_episodes):
    new_data = np.random.rand(10)
    reward = torch.max(classificador(torch.FloatTensor(new_data))).item()
    next_state, done = np.random.rand(10), np.random.rand() < 0.1
    action = select_action(new_data, epsilon)
    update_model(new_data, action, reward, next_state, done) 
    print(f'Episódio {episode}, Ação: {action}, Recompensa: {reward}')
torch.save(policy_net.state_dict(), "novo_modelo_reforco.pth")
