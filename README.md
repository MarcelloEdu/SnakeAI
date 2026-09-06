# Snake AI - Meu Primeiro Projeto de Machine Learning

Este repositório contém o meu primeiro projeto prático explorando o mundo do **Machine Learning**. O objetivo foi ensinar um computador a jogar o clássico "Jogo da Cobrinha" (Snake) do zero, sem programar nenhuma regra de "como vencer". Em vez disso, a máquina aprende sozinha através de tentativa e erro usando **Aprendizado por Reforço**.

<p align="center">
    <img src = "assets/learning.MOV">
</p>

---

## Como a Máquina Aprende?

Como este é meu projeto de introdução à área, documentei aqui os conceitos centrais que fizeram a mágica acontecer.

O aprendizado ocorre em um ciclo infinito de 3 passos: **Estado, Ação e Recompensa**.

### 1- O Estado (O que a cobra vê)

O ambiente traduz o jogo em um array de 11 valores booleanos (0 ou 1) que funcionam como os "sensores" da cobra:

* Perigo à frente, à direita ou à esquerda? (3 sensores)
* Qual a direção atual? (Cima, Baixo, Esquerda, Direita - 4 sensores)
* Onde está a comida em relação à cabeça? (Acima, Abaixo, Esquerda, Direita - 4 sensores)

### 2- A Ação (O que a cobra faz)

Com base nesses 11 números, a Rede Neural toma uma decisão. Para evitar que a IA inverta o movimento e morra instantaneamente, a ação é sempre relativa à própria cabeça:

* `[1, 0, 0]` - Seguir em frente
* `[0, 1, 0]` - Virar à direita
* `[0, 0, 1]` - Virar à esquerda

### 3. A Recompensa (O Feedback)

O jogo responde à ação com um sistema de pontuação estrito, dizendo à IA se ela foi bem ou mal:

* **+10 pontos** se comer a maçã.
* **-10 pontos** se bater na parede ou no próprio corpo.
* **0 pontos** se apenas der um passo seguro.

---

## 📘 Anotações

Durante o desenvolvimento, me deparei com comportamentos curiosos que me ensinaram conceitos de Machine Learning:

* **Exploração vs. Explotação (O Epsilon):**
No começo, a rede neural nasce "burra" e precisa jogar de forma 100% aleatória para descobrir que a maçã dá pontos e a parede mata. Isso se chama *Exploração*. Com o tempo, a aleatoriedade (controlada pela variável `epsilon`) diminui, e a IA passa a usar a inteligência que adquiriu. Se o *epsilon* for removido cedo demais, a IA pode ficar "traumatizada" e, por exemplo, andar apenas para a direita infinitamente por medo de virar (aconteceu durante os primeiros testes).
* **A Equação de Bellman e o Q-Value:**
A máquina não aprende apenas com a recompensa imediata. Quando ela come a maçã, o algoritmo (Deep Q-Network) atualiza os pesos da rede neural espalhando o valor daquela maçã para as ações anteriores que a levaram até lá. É assim que ela aprende a planejar rotas.
* **Memória de Longo Prazo:**
No fim de cada partida, a IA não treina apenas com o último jogo. Ela puxa um lote aleatório de 1000 jogadas de sua memória e treina tudo de novo. Isso evita que ela "esqueça" como desviar de paredes simples só porque passou as últimas 10 partidas focada em manobras complexas.
* **O Platô de Convergência:**
Ao redor de 30 pontos, o gráfico de aprendizado costuma estabilizar (log n). Isso ocorre porque a cobra fica muito grande 😏 e os 11 sensores de 1px não são mais suficientes para ela entender o labirinto que seu próprio corpo formou (por isso é dificil nesse modelo inicial a IA entender que ela deve por exemplo planejar o espaço de saida antes de comer a maçã).
* **Futuro**
O modelo atual esbarra em um limite ao redor de 30 pontos porque a visão da cobra é limitada (seus 11 sensores olham apenas um quadrado de distância). Ela age como um algoritmo guloso e, ao crescer, acaba se encurralando no próprio corpo por não ter visão global do tabuleiro.

Como este é meu primeiro projeto em ML, mapeei as soluções abaixo como opçÕes para o futuro:
Visão Computacional: Em vez de 11 sensores locais, o estado passaria a ser a matriz completa do jogo (como uma imagem). A rede aprenderia padrões espaciais complexos para não se enrolar, mas exigirá aprender a lidar com um alto custo computacional($).
Busca + ML: Unir algoritmos de caminho (como Backtracking) com Redes Neurais. O objetivo seria estudar Monte Carlo Tree Search (MCTS) para simular o futuro em uma árvore de decisão e usar a IA para "podar" os caminhos ruins, planejando rotas de fuga.
---

## Arquitetura do Projeto

O código está dividido em 4 módulos principais para separar as responsabilidades:

1. `snakeAI.py`: O ambiente criado no Pygame. Não possui loop próprio, apenas reage aos comandos recebidos e devolve as recompensas.
2. `agente.py`: A ponte lógica. Coleta os 11 sensores do jogo, guarda as jogadas na memória e gerencia a transição de movimentos aleatórios para movimentos inteligentes.
3. `modelo.py`: O cérebro em PyTorch. Uma rede neural simples (Linear_QNet) e o otimizador que aplica a matemática de correção de erros.
4. `helper.py`: Responsável por plotar o gráfico de desempenho usando Matplotlib em tempo real.

---

## Como Rodar na sua Máquina

tenha o Python instalado. É recomendado criar um ambiente virtual (venv).

```bash
pip install pygame-ce torch torchvision numpy matplotlib

```

*(Nota: a lib `pygame-ce` é a Community Edition, melhor otimizada para chips como Apple Silicon (Processador da minha maquina)).*

### Para Treinar a IA

Execute o arquivo principal. Uma janela com o jogo e outra com o gráfico se abrirão.

```bash
python main.py

```

*O modelo salvará os melhores pesos automaticamente na pasta `/model/model.pth` sempre que bater um novo recorde.*
```