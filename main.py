import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import networkx as nx
import matplotlib.pyplot as plt
from ad_database import ads_df, get_ad_by_id
from q_learning_agent import AdRecommendationAgent
from user_simulator_main import UserSimulator
from viper_graph import draw_viper_tree


def visualize_q_table(agent, user_id, ad_database, top_n=10):
    if user_id not in agent.user_q_tables:
        print(f"User {user_id} not found in Q-tables")
        return

    q_table = agent.user_q_tables[user_id]
    state = agent.user_states[user_id]

    # In bảng Q-table của state hiện tại
    q_df = pd.DataFrame([
        {
            'ad_id': ad_id,
            'Q_value': float(q_table[state, ad_id]),
            'product': ad_database.loc[ad_id]['product'],
            'category': ad_database.loc[ad_id]['category']
        }
        for ad_id in range(q_table.shape[1])
    ])
    q_df = q_df.sort_values(by='Q_value', ascending=False).reset_index(drop=True)
    print(f"\n🧠 Q-table cho User {user_id} (Top {top_n}, State {state}):")
    print(q_df.head(top_n).to_string(index=False))

    #Viper
    draw_viper_tree(q_table, ad_database, user_id=user_id)







def run_simulation(num_users=5, num_iterations=1000):
    """Run the full simulation"""
    num_ads = len(ads_df)
    num_categories = len(ads_df['category'].unique())
    
    agent = AdRecommendationAgent(
        num_ads=num_ads,
        num_categories=num_categories,
        alpha=0.1,  
        gamma=0.8,  
        epsilon=0.2  
    )
    
    simulator = UserSimulator(num_users=num_users)
    
    total_rewards = np.zeros((num_users, num_iterations))
    click_rates = np.zeros((num_users, num_iterations))
    
    for iteration in range(num_iterations):
        if iteration % 100 == 0 and iteration > 0:
            agent.epsilon = max(0.05, agent.epsilon * 0.9)
            
        for user_id in range(num_users):
            user_history = simulator.get_user_history(user_id)
            ad_id = agent.select_ad(user_id, user_history)
            clicked, reward = simulator.simulate_interaction(user_id, ad_id)
            agent.update_q_table(user_id, ad_id, reward, simulator.get_user_history(user_id))
            
            total_rewards[user_id, iteration] = reward
            click_rates[user_id, iteration] = clicked
            
        if iteration % 100 == 0:
            avg_reward = np.mean(total_rewards[:, max(0, iteration-99):iteration+1])
            avg_click = np.mean(click_rates[:, max(0, iteration-99):iteration+1])
            print(f"Iteration {iteration}: Avg Reward: {avg_reward:.4f}, Click Rate: {avg_click:.4f}")
    
    return {
        'agent': agent,
        'simulator': simulator,
        'total_rewards': total_rewards,
        'click_rates': click_rates
    }

def plot_results(results):
    """Plot simulation results"""
    num_users = results['total_rewards'].shape[0]
    num_iterations = results['total_rewards'].shape[1]
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(results['click_rates'], axis=0))
    plt.title('Average Click Rate Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Click Rate')
    plt.savefig('learning_results.png')
    plt.show()
    
if __name__ == "__main__":
    print("Starting personalized ad recommendation simulation...")
    start_time = time.time()
    
    results = run_simulation(num_users=5, num_iterations=1000)
    
    print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds")
    plot_results(results)
    
    for user_id in range(3):  # Show VIPER for first 3 users
        visualize_q_table(results['agent'], user_id, ads_df)
