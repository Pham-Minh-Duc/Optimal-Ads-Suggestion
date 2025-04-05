import numpy as np
import pandas as pd
from ad_database import get_ad_by_id, categories

class UserSimulator:
    def __init__(self, num_users=10):
        """
        Simulate user behavior for testing the recommendation system
        
        Parameters:
        - num_users: Number of simulated users
        """
        self.num_users = num_users
        self.user_preferences = {}
        self.user_history = {}
        
        # Initialize user preferences
        self._initialize_users()
        
    def _initialize_users(self):
        """Initialize user preferences for each category"""
        category_names = list(categories.keys())
        
        for user_id in range(self.num_users):
            # Random preference for each category (values between 0.1 and 1.0)
            preferences = {}
            
            # Each user has 1-2 favorite categories
            favorite_categories = np.random.choice(
                category_names, 
                size=np.random.randint(1, 3), 
                replace=False
            )
            
            for category in category_names:
                if category in favorite_categories:
                    # Higher preference for favorite categories
                    preferences[category] = np.random.uniform(0.6, 1.0)
                else:
                    # Lower preference for other categories
                    preferences[category] = np.random.uniform(0.1, 0.3)
            
            self.user_preferences[user_id] = preferences
            self.user_history[user_id] = []
            
    def simulate_interaction(self, user_id, ad_id):
        """
        Simulate user interaction with an ad
        
        Returns:
        - clicked: Boolean indicating if user clicked
        - reward: Reward value (1 for click, 0 for no click)
        """
        ad = get_ad_by_id(ad_id)
        if ad is None:
            return False, 0
            
        category = ad['category']
        preference = self.user_preferences[user_id].get(category, 0.1)
        
        # Base click probability from ad combined with user preference
        click_probability = ad['click_rate_base'] * preference * 5
        
        # Cap probability at 0.95
        click_probability = min(click_probability, 0.95)
        
        # Determine if user clicks
        clicked = np.random.random() < click_probability
        
        # Record interaction in user history
        self.user_history[user_id].append({
            'ad_id': ad_id,
            'category': category,
            'product': ad['product'],
            'clicked': clicked
        })
        
        # Return click status and reward (1 for click, 0 for no click)
        return clicked, 1 if clicked else 0
        
    def get_user_history(self, user_id):
        """Get user interaction history"""
        return self.user_history.get(user_id, [])
        
    def print_user_preferences(self, user_id):
        """Print user preferences for debugging"""
        if user_id in self.user_preferences:
            print(f"User {user_id} preferences:")
            for category, pref in self.user_preferences[user_id].items():
                print(f"  {category}: {pref:.2f}")
        else:
            print(f"User {user_id} not found")