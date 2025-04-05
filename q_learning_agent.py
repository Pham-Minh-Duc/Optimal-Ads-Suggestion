import numpy as np
import pandas as pd
from ad_database import get_ads_by_category, ads_df

class AdRecommendationAgent:
    def __init__(self, num_ads, num_categories, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize the Q-learning agent
        
        Parameters:
        - num_ads: Total number of ads
        - num_categories: Total number of categories
        - alpha: Learning rate
        - gamma: Discount factor
        - epsilon: Exploration rate
        """
        self.num_ads = num_ads
        self.num_categories = num_categories
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table for each user
        self.user_q_tables = {}
        self.user_states = {}
        self.categories = list(set(ads_df['category']))
        
    def get_state_index(self, user_history):
        """Convert user history to a state index"""
        if not user_history:
            return 0  # Default state
        
        # Get most frequent category from recent history
        recent_history = user_history[-5:] if len(user_history) > 5 else user_history
        categories = [item['category'] for item in recent_history if 'category' in item]
        if not categories:
            return 0
            
        from collections import Counter
        most_common = Counter(categories).most_common(1)[0][0]
        return self.categories.index(most_common) + 1
    
    def initialize_user(self, user_id):
        """Initialize Q-table for a new user"""
        # States: 0 (no history) + 1 state per category
        num_states = 1 + self.num_categories
        
        # Actions: One per ad
        num_actions = self.num_ads
        
        # Initialize Q-table with small random values
        self.user_q_tables[user_id] = np.random.uniform(0, 0.01, (num_states, num_actions))
        self.user_states[user_id] = 0  # Initial state
        
    def select_ad(self, user_id, user_history=None):
        """Select an ad for the user using epsilon-greedy policy"""
        # Initialize user if new
        if user_id not in self.user_q_tables:
            self.initialize_user(user_id)
            
        # Update state based on user history
        if user_history:
            self.user_states[user_id] = self.get_state_index(user_history)
            
        state = self.user_states[user_id]
        
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            # Exploration: select random ad
            ad_id = np.random.randint(0, self.num_ads)
        else:
            # Exploitation: select best ad according to Q-table
            ad_id = np.argmax(self.user_q_tables[user_id][state])
            
        return ad_id
    
    def update_q_table(self, user_id, ad_id, reward, new_history=None):
        """Update Q-table based on reward"""
        if user_id not in self.user_q_tables:
            return
            
        old_state = self.user_states[user_id]
        
        # Update state if new history provided
        if new_history:
            new_state = self.get_state_index(new_history)
            self.user_states[user_id] = new_state
        else:
            new_state = old_state
            
        # Q-learning update formula
        old_value = self.user_q_tables[user_id][old_state, ad_id]
        next_max = np.max(self.user_q_tables[user_id][new_state])
        
        # Update Q-value
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.user_q_tables[user_id][old_state, ad_id] = new_value
        
    def get_top_ads_for_user(self, user_id, n=10):
        """Get top n ads for a user based on Q-values"""
        if user_id not in self.user_q_tables:
            self.initialize_user(user_id)
            
        state = self.user_states[user_id]
        q_values = self.user_q_tables[user_id][state]
        
        # Get indices of top n ads
        top_ad_indices = q_values.argsort()[-n:][::-1]
        return top_ad_indices.tolist()
    
    def get_preference_based_ads(self, user_id, user_preferences, n=10):
        """
        Get ads based on user preferences, prioritizing ads from categories with highest preferences
        
        Parameters:
        - user_id: User ID
        - user_preferences: Dictionary of user preferences by category
        - n: Number of ads to return
        
        Returns:
        - List of ad IDs
        """
        # Sort categories by preference (highest first)
        sorted_categories = sorted(user_preferences.items(), key=lambda x: x[1], reverse=True)
        
        # Get ads for each category based on Q-values
        recommended_ads = []
        ads_per_category = {}
        
        # First, get Q-values for all ads
        if user_id not in self.user_q_tables:
            self.initialize_user(user_id)
            
        state = self.user_states[user_id]
        q_values = self.user_q_tables[user_id][state]
        
        # For each category, get top ads based on Q-values
        for category, preference in sorted_categories:
            # Get all ads in this category
            category_ads = ads_df[ads_df['category'] == category]['ad_id'].values
            
            if len(category_ads) > 0:
                # Get Q-values for these ads
                category_q_values = q_values[category_ads]
                
                # Sort ads by Q-value
                sorted_indices = category_q_values.argsort()[::-1]
                top_category_ads = category_ads[sorted_indices]
                
                # Store top ads for this category
                ads_per_category[category] = top_category_ads.tolist()
        
        # Now build the final recommendation list
        # First, include at least one ad from each category with preference > 0.1
        for category, preference in sorted_categories:
            if preference > 0.1 and category in ads_per_category and len(ads_per_category[category]) > 0:
                recommended_ads.append(ads_per_category[category][0])
                ads_per_category[category] = ads_per_category[category][1:]  # Remove the used ad
                
                if len(recommended_ads) >= n:
                    break
        
        # Then fill remaining slots with ads from highest preference categories
        remaining_slots = n - len(recommended_ads)
        if remaining_slots > 0:
            for category, preference in sorted_categories:
                if category in ads_per_category:
                    # Add remaining ads from this category
                    category_ads_to_add = ads_per_category[category][:remaining_slots]
                    recommended_ads.extend(category_ads_to_add)
                    remaining_slots -= len(category_ads_to_add)
                    
                    if remaining_slots <= 0:
                        break
        
        # If we still don't have enough ads, add some based on pure Q-values
        if len(recommended_ads) < n:
            # Get top ads by Q-value that aren't already in the list
            top_q_ads = q_values.argsort()[::-1]
            for ad_id in top_q_ads:
                if ad_id not in recommended_ads:
                    recommended_ads.append(ad_id)
                    if len(recommended_ads) >= n:
                        break
        
        return recommended_ads[:n]  # Ensure we return exactly n ads