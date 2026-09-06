from agente import Agent
from snakeAI import SnakeGameAI
from helper import plot

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 30
    
    
    agent = Agent()
    game = SnakeGameAI()
    
    print("Iniciando o treinamento! Pressione Ctrl+C no terminal para parar.")
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        agent.trainer.train_step(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print(f'Jogo: {agent.n_games} | Score: {score} | Recorde: {record}')
            
            # --- LÓGICA DO GRÁFICO AQUI ---
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()