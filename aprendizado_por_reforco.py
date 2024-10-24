import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError
import os
from ultralytics import YOLO
import torch.nn.functional as F

# Transformação para pré-processar as imagens
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Dimensões compatíveis com YOLOv8
    transforms.ToTensor()            # Converte a imagem para tensor
])

# Dataset personalizado que carrega imagens com múltiplas classes
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")  # Carregar a imagem e converter para RGB
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except UnidentifiedImageError:
            # Caso o arquivo não seja uma imagem, ignorar e retornar None
            return None, None

# Caminho para o diretório de imagens
image_dir = 'C:\\Users\\Administrador\\Desktop\\VisEdu\\Aprendizado_por_reforco_yolo\\train'

# Carregar dataset com imagens sem divisão por classes
dataset = CustomImageDataset(image_dir=image_dir, transform=transform)

# Remover itens 'None' ao carregar o DataLoader
def collate_fn(batch):
    # Filtrar os casos onde image ou img_path é None
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    images, img_paths = zip(*batch)
    return torch.stack(images), img_paths

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Carregar o modelo YOLOv8 pré-treinado
yolo_model = YOLO("best.pt")

# Definindo a arquitetura da rede
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Definir as camadas convolucionais
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),  # Camada convolucional 1
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),  # Camada convolucional 2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),  # Camada convolucional 3
            nn.ReLU()
        )
        # Camada totalmente conectada (ajustar este número com base na saída convolucional)
        self.fc = nn.Sequential(
            nn.Linear(64 * 77 * 77, 128),  # Ajuste este valor baseado na saída da convolução
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 ações
        )

    def forward(self, x):
        x = self.conv(x)          # Passa pela parte convolucional
        print("Saída da convolução:", x.shape)  # Debug: imprimir o tamanho
        x = torch.flatten(x, start_dim=1)  # Aplana a saída da convolução
        x = self.fc(x)            # Passa pela parte totalmente conectada
        return x

# Inicializando a rede de políticas
policy_net = PolicyNetwork()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Hiperparâmetros
gamma, epsilon, num_episodes = 0.99, 0.1, 100

# Função para calcular a recompensa baseada nas detecções do YOLOv8
def calcular_recompensa(deteccoes, classes_esperadas):
    recompensa = 0
    # Extrair as caixas detectadas e as classes preditas
    for detecao in deteccoes[0].boxes:  # Acessa o primeiro elemento de detecções
        classe_predita = int(detecao.cls)  # Acessa a classe predita
        if classe_predita in classes_esperadas:
            recompensa += 1  # Recompensa por detecção correta
        else:
            recompensa -= 1  # Penalidade por detecção incorreta
    return recompensa

# Função de atualização do modelo
def update_model(state, action, reward, next_state, done):
    q_value = policy_net(state).gather(1, torch.LongTensor([[action]]))
    with torch.no_grad():
        max_next_q = policy_net(next_state).max(1)[0]
        target = reward + gamma * max_next_q * (1 - done)
    optimizer.zero_grad()
    loss = criterion(q_value, target.unsqueeze(1))
    loss.backward()
    optimizer.step()

# Função para escolher a ação (política epsilon-greedy)
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 2)  # Número de ações possíveis
    else:
        with torch.no_grad():
            return policy_net(state).argmax().item()

# Loop de treinamento usando imagens
for episode in range(num_episodes):
    total_images = len(dataset)  # Total de imagens no dataset
    for idx, (images, img_paths) in enumerate(data_loader):
        if images is None or img_paths is None:
            continue  # Ignorar se a imagem não for válida
        
        # Exibir o progresso
        print(f'Episódio {episode}, Processando imagem {idx + 1} de {total_images}: {img_paths[0]}')

        # Usar a imagem carregada como o estado atual
        state = images
        
        # Extrair a recompensa do YOLOv8
        deteccoes = yolo_model(state)  # Faz a detecção usando o YOLO
        classes_esperadas = [0, 1, 2, 3, 4, 5]  # dormindo, prestando atenção, mexendo no celular, copiando, disperso, trabalho em grupo
        reward = calcular_recompensa(deteccoes, classes_esperadas)
        
        next_state = torch.rand_like(state)  # Usando um next_state aleatório para este exemplo
        done = np.random.rand() < 0.1  # Define se a episódio está concluído aleatoriamente
        action = select_action(state, epsilon)
        
        update_model(state, action, reward, next_state, done)
        print(f'Ação: {action}, Recompensa: {reward}')
        
# Salvar o estado da rede neural treinada
yolo_model.save('new.pt')
