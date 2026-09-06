import matplotlib.pyplot as plt

# Ativa o modo interativo do Matplotlib para atualizar o gráfico em tempo real
plt.ion() 

def plot(scores, mean_scores):
    plt.clf() # Limpa o gráfico anterior
    plt.title('Progresso do Treinamento')
    plt.xlabel('Número de Jogos')
    plt.ylabel('Pontuação')
    
    # Plota as duas linhas
    plt.plot(scores, label='Score Atual', color='blue')
    plt.plot(mean_scores, label='Média (Histórico)', color='orange')
    
    plt.ylim(ymin=0)
    
    # Escreve o número exato no final de cada linha
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    
    plt.legend(loc='upper left')
    
    # Atualiza a janela
    plt.show(block=False)
    plt.pause(.1)