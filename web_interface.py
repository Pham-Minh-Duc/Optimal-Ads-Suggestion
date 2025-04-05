import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from ad_database import ads_df, get_ad_by_id
from q_learning_agent import AdRecommendationAgent
from user_simulator import UserSimulator

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.agent = AdRecommendationAgent(
        num_ads=len(ads_df),
        num_categories=len(ads_df['category'].unique()),
        alpha=0.1,
        gamma=0.8,
        epsilon=0.1
    )
    st.session_state.simulator = UserSimulator(num_users=10)
    st.session_state.current_user = 0
    st.session_state.clicks = []
    st.session_state.rewards = []
    st.session_state.categories = []
    st.session_state.current_page = "home"
    st.session_state.current_ad = None
    st.session_state.interaction_count = 0
    st.session_state.last_category_update = {}  # Theo dõi thời gian cập nhật preference
    st.session_state.category_avg_rewards = {}  # Lưu trữ reward trung bình cho mỗi danh mục
    st.session_state.initialized = True

# App title
st.title("Personalized Ad System with Q-Learning")

# Sidebar for user selection
st.sidebar.header("User Settings")
user_id = st.sidebar.selectbox(
    "Select User",
    options=list(range(10)),
    index=st.session_state.current_user
)

if user_id != st.session_state.current_user:
    st.session_state.current_user = user_id
    st.session_state.clicks = []
    st.session_state.rewards = []
    st.session_state.categories = []

# Display user preferences
st.sidebar.subheader("User Preferences (Learned)")
user_prefs = st.session_state.simulator.user_preferences[user_id]
for category, pref in user_prefs.items():
    st.sidebar.progress(pref)
    st.sidebar.text(f"{category}: {pref:.2f}")

# Function to create a unique key for each button
def get_unique_key(prefix, ad_id):
    return f"{prefix}_{ad_id}_{int(time.time() * 1000)}"

# Function to calculate reward based on probabilities
def calculate_reward(user_id, ad_id, action_type):
    """
    Tính toán reward dựa trên xác suất và hành động
    
    Parameters:
    - user_id: ID của người dùng
    - ad_id: ID của quảng cáo
    - action_type: Loại hành động ('view' hoặc 'purchase')
    
    Returns:
    - reward: Giá trị phần thưởng
    """
    ad = get_ad_by_id(ad_id)
    category = ad['category']
    
    # Lấy sở thích người dùng cho danh mục này
    user_preference = st.session_state.simulator.user_preferences[user_id].get(category, 0.0)
    
    # Lấy tỷ lệ click cơ bản của quảng cáo
    base_click_rate = ad.get('click_rate_base', 0.03)  # Mặc định 3%
    
    # Tính xác suất click dựa trên sở thích người dùng và tỷ lệ cơ bản
    click_probability = base_click_rate * (1 + 2 * user_preference)  # Tăng tỷ lệ theo sở thích
    
    # Giới hạn xác suất trong khoảng [0.01, 0.3]
    click_probability = max(0.01, min(0.3, click_probability))
    
    # Tính xác suất mua hàng (thường thấp hơn xác suất click)
    purchase_probability = click_probability * 0.2  # 20% người click sẽ mua
    
    if action_type == 'view':
        # Reward cho việc xem là xác suất click
        return click_probability
    elif action_type == 'purchase':
        # Reward cho việc mua hàng cao hơn
        return 0.5 + purchase_probability
    else:
        return 0

# Function to update category average rewards
def update_category_avg_reward(user_id, category, reward):
    """
    Cập nhật reward trung bình cho danh mục
    
    Parameters:
    - user_id: ID của người dùng
    - category: Danh mục
    - reward: Phần thưởng mới
    """
    if user_id not in st.session_state.category_avg_rewards:
        st.session_state.category_avg_rewards[user_id] = {}
        
    if category not in st.session_state.category_avg_rewards[user_id]:
        st.session_state.category_avg_rewards[user_id][category] = {'total': 0, 'count': 0}
        
    st.session_state.category_avg_rewards[user_id][category]['total'] += reward
    st.session_state.category_avg_rewards[user_id][category]['count'] += 1

# Function to get category average reward
def get_category_avg_reward(user_id, category):
    """
    Lấy reward trung bình cho danh mục
    
    Parameters:
    - user_id: ID của người dùng
    - category: Danh mục
    
    Returns:
    - avg_reward: Reward trung bình
    """
    if user_id not in st.session_state.category_avg_rewards:
        return 0
        
    if category not in st.session_state.category_avg_rewards[user_id]:
        return 0
        
    cat_data = st.session_state.category_avg_rewards[user_id][category]
    if cat_data['count'] > 0:
        return cat_data['total'] / cat_data['count']
    return 0

# Modified handle_ad_click function to use probability-based rewards
def handle_ad_click(ad_id):
    # Save ad_id to session state
    st.session_state.current_ad = ad_id
    st.session_state.current_page = "ad_detail"

    user_id = st.session_state.current_user
    
    # Calculate reward based on click probability
    reward = calculate_reward(user_id, ad_id, 'view')
    
    # Get ad information
    ad = get_ad_by_id(ad_id)
    category = ad['category']
    
    # Update user preferences based on reward
    current_pref = st.session_state.simulator.user_preferences[user_id].get(category, 0.0)
    
    # Use reward to adjust update rate
    update_rate = 0.05 * (1 + reward)  # Tốc độ cập nhật tăng theo reward
    
    new_pref = current_pref + update_rate * (1.0 - current_pref)
    new_pref = max(0.0, min(1.0, new_pref))
    st.session_state.simulator.user_preferences[user_id][category] = new_pref
    
    # Record interaction in user history
    if user_id not in st.session_state.simulator.user_history:
        st.session_state.simulator.user_history[user_id] = []
        
    st.session_state.simulator.user_history[user_id].append({
        'ad_id': ad_id,
        'category': category,
        'product': ad['product'],
        'action': 'view',
        'reward': reward,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Update category average reward
    update_category_avg_reward(user_id, category, reward)

    # Update Q-table with new reward
    st.session_state.agent.update_q_table(
        user_id,
        ad_id,
        reward,
        st.session_state.simulator.get_user_history(user_id)
    )

    # Record metrics
    st.session_state.clicks.append(True)
    st.session_state.rewards.append(reward)
    
    # Update category and timestamp
    st.session_state.categories.append(category)
    st.session_state.last_category_update[category] = time.time()
    
    st.session_state.interaction_count += 1

    # Force a rerun to update the UI
    st.rerun()

# Function to handle purchase
def handle_purchase(ad_id):
    user_id = st.session_state.current_user
    
    # Calculate reward for purchase
    purchase_reward = calculate_reward(user_id, ad_id, 'purchase')
    
    # Get ad information
    ad = get_ad_by_id(ad_id)
    category = ad['category']
    
    # Update user preferences based on purchase reward
    current_pref = st.session_state.simulator.user_preferences[user_id].get(category, 0.0)
    
    # Stronger update for purchase
    update_rate = 0.1 * (1 + purchase_reward)
    new_pref = current_pref + update_rate * (1.0 - current_pref)
    new_pref = max(0.0, min(1.0, new_pref))
    st.session_state.simulator.user_preferences[user_id][category] = new_pref
    
    # Record purchase in user history
    st.session_state.simulator.user_history[user_id].append({
        'ad_id': ad_id,
        'category': category,
        'product': ad['product'],
        'action': 'purchase',
        'reward': purchase_reward,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Update category average reward
    update_category_avg_reward(user_id, category, purchase_reward)
    
    # Update Q-table with purchase reward
    st.session_state.agent.update_q_table(
        user_id,
        ad_id,
        purchase_reward,
        st.session_state.simulator.get_user_history(user_id)
    )
    
    # Update rewards list with the purchase reward
    if len(st.session_state.rewards) > 0:
        st.session_state.rewards[-1] = purchase_reward

# Function to go back to home
def go_back_to_home():
    st.session_state.current_page = "home"
    st.session_state.current_ad = None  # Reset current ad
    st.rerun()

# Modified function to get recommendations using reward information
def get_recommendations(user_id, user_prefs, n=30):
    all_categories = ads_df['category'].unique()
    
    # Nếu chưa có đủ tương tác (ví dụ: dưới 5 lần), trả về ads đa dạng
    if st.session_state.interaction_count < 5:
        recommended_ads = []
        
        # Đảm bảo có đại diện từ mỗi category
        for category in all_categories:
            category_ads = ads_df[ads_df['category'] == category]
            if len(category_ads) > 0:
                # Lấy 2-3 ads từ mỗi category để đảm bảo đa dạng
                num_ads = min(3, len(category_ads))
                category_sample = category_ads['ad_id'].sample(num_ads).tolist()
                recommended_ads.extend(category_sample)
        
        # Nếu chưa đủ n ads, bổ sung random
        remaining_slots = n - len(recommended_ads)
        if remaining_slots > 0:
            remaining_ads = ads_df[~ads_df['ad_id'].isin(recommended_ads)]
            if len(remaining_ads) > 0:
                random_fill = remaining_ads['ad_id'].sample(min(remaining_slots, len(remaining_ads))).tolist()
                recommended_ads.extend(random_fill)
        
        # Trộn ngẫu nhiên để không có pattern cố định
        random.shuffle(recommended_ads)
        return recommended_ads[:n]

    # Sau khi có đủ tương tác, kết hợp đa dạng và ưu tiên theo preference và reward
    recommended_ads = []
    
    # Sắp xếp categories theo preference, reward và thời gian cập nhật
    categories_with_data = []
    for category in all_categories:
        pref = user_prefs.get(category, 0.0)
        avg_reward = get_category_avg_reward(user_id, category)
        last_update = st.session_state.last_category_update.get(category, 0)
        
        # Điều chỉnh preference dựa trên reward
        adjusted_pref = pref * (1 + avg_reward)
        
        categories_with_data.append((category, adjusted_pref, avg_reward, last_update))
    
    # Sắp xếp theo preference đã điều chỉnh (cao nhất trước)
    sorted_categories = sorted(categories_with_data, 
                              key=lambda x: (x[1], x[3]), 
                              reverse=True)
    
    # BƯỚC 1: Đảm bảo đa dạng - mỗi category có ít nhất 2 ads
    min_ads_per_category = 2
    diversity_ads = []
    
    for category, _, _, _ in sorted_categories:
        category_ads = ads_df[ads_df['category'] == category]
        if len(category_ads) == 0:
            continue
            
        # Ưu tiên lấy theo Q-values cho mỗi category
        category_ad_ids = category_ads['ad_id'].tolist()
        top_q_ads = [
            ad_id for ad_id in st.session_state.agent.get_top_ads_for_user(user_id, n=len(category_ad_ids))
            if ad_id in category_ad_ids
        ]
        
        # Lấy ít nhất min_ads_per_category ads từ mỗi category
        num_ads = min(min_ads_per_category, len(category_ads))
        if len(top_q_ads) >= num_ads:
            diversity_ads.extend(top_q_ads[:num_ads])
        else:
            # Nếu không đủ từ Q-values, bổ sung random
            diversity_ads.extend(top_q_ads)
            remaining = num_ads - len(top_q_ads)
            remaining_ads = [ad for ad in category_ad_ids if ad not in top_q_ads]
            if remaining_ads:
                diversity_ads.extend(random.sample(remaining_ads, min(remaining, len(remaining_ads))))
    
    # BƯỚC 2: Phân bổ số slots còn lại theo preference đã điều chỉnh
    remaining_slots = n - len(diversity_ads)
    
    if remaining_slots > 0:
        # Tính tổng preference đã điều chỉnh để phân bổ tỷ lệ
        total_adjusted_pref = sum(pref for _, pref, _, _ in sorted_categories) or 1  # Tránh chia cho 0
        
        # Giới hạn tối đa số ads cho mỗi category
        max_additional_per_category = max(3, remaining_slots // 2)
        
        # Phân bổ số lượng ads cho mỗi category dựa trên preference đã điều chỉnh
        category_allocation = {}
        remaining_after_allocation = remaining_slots
        
        for category, adjusted_pref, avg_reward, _ in sorted_categories:
            if adjusted_pref > 0:
                # Phân bổ số lượng ads tỷ lệ với preference đã điều chỉnh
                raw_allocation = int((adjusted_pref / total_adjusted_pref) * remaining_slots)
                
                # Giới hạn không vượt quá max_additional_per_category
                allocation = min(raw_allocation, max_additional_per_category)
                
                category_allocation[category] = allocation
                remaining_after_allocation -= allocation
        
        # Nếu còn slots chưa phân bổ, phân phối đều cho các category có preference thấp hơn
        if remaining_after_allocation > 0:
            # Ưu tiên các category có preference thấp hơn
            low_pref_categories = [(cat, pref) for cat, pref, _, _ in sorted_categories if pref < 0.3]
            
            if low_pref_categories:
                # Sắp xếp theo preference tăng dần để ưu tiên category có preference thấp
                low_pref_categories.sort(key=lambda x: x[1])
                
                # Phân bổ thêm cho các category có preference thấp
                slots_per_category = remaining_after_allocation // len(low_pref_categories) or 1
                for category, _ in low_pref_categories:
                    additional = min(slots_per_category, remaining_after_allocation)
                    category_allocation[category] = category_allocation.get(category, 0) + additional
                    remaining_after_allocation -= additional
                    if remaining_after_allocation <= 0:
                        break
            
            # Nếu vẫn còn slots, phân bổ cho category có preference cao nhất
            if remaining_after_allocation > 0 and sorted_categories:
                top_category = sorted_categories[0][0]
                category_allocation[top_category] = category_allocation.get(top_category, 0) + remaining_after_allocation
        
        # BƯỚC 3: Lấy ads cho từng category theo thứ tự ưu tiên
        for category, _, _, _ in sorted_categories:
            if category not in category_allocation or category_allocation[category] <= 0:
                continue
                
            # Lấy ads thuộc category này (loại bỏ ads đã chọn)
            category_ads = ads_df[ads_df['category'] == category]
            category_ads = category_ads[~category_ads['ad_id'].isin(diversity_ads)]
            
            if len(category_ads) == 0:
                continue
                
            # Lấy top ads từ Q-table cho category này
            all_ads_for_category = category_ads['ad_id'].tolist()
            top_q_ads_for_category = [
                ad_id for ad_id in st.session_state.agent.get_top_ads_for_user(user_id, n=len(all_ads_for_category))
                if ad_id in all_ads_for_category and ad_id not in recommended_ads
            ]
            
            # Nếu không đủ ads từ Q-table, bổ sung ngẫu nhiên
            if len(top_q_ads_for_category) < category_allocation[category]:
                remaining_ads = [
                    ad_id for ad_id in all_ads_for_category 
                    if ad_id not in top_q_ads_for_category and ad_id not in recommended_ads
                ]
                random.shuffle(remaining_ads)
                top_q_ads_for_category.extend(remaining_ads)
            
            # Lấy số lượng ads cần thiết
            selected_ads = top_q_ads_for_category[:category_allocation[category]]
            recommended_ads.extend(selected_ads)
    
    # Kết hợp ads đa dạng và ads theo preference
    final_recommendations = diversity_ads + recommended_ads
    
    # BƯỚC 4: Nếu vẫn chưa đủ n ads, bổ sung random
    if len(final_recommendations) < n:
        remaining = n - len(final_recommendations)
        available_ads = ads_df[~ads_df['ad_id'].isin(final_recommendations)]
        if len(available_ads) > 0:
            random_fill = available_ads['ad_id'].sample(min(remaining, len(available_ads))).tolist()
            final_recommendations.extend(random_fill)
    
    return final_recommendations[:n]

# Check which page to display
if st.session_state.current_page == "home":
    # Main content - Home page
    st.header("Ad Recommendations")

    # Get recommended ads based on user preferences
    user_history = st.session_state.simulator.get_user_history(user_id)
    user_prefs = st.session_state.simulator.user_preferences[user_id]

    # Use custom recommendation function
    all_ad_ids = get_recommendations(user_id, user_prefs, n=30)

    # Display ads - Modified to show 30 ads in 6 rows of 5
    st.subheader("Top Recommendations")

    # Display 6 rows of 5 ads each
    for row in range(6):
        cols = st.columns(5)
        for i, ad_id in enumerate(all_ad_ids[row*5:(row+1)*5]):
            ad = get_ad_by_id(ad_id)
            with cols[i]:
                st.subheader(ad['title'])
                st.text(f"Category: {ad['category']}")
                st.text(f"Product: {ad['product']}")
                
                button_key = get_unique_key("ad", ad_id)
                if st.button(f"View {row*5+i+1}", key=button_key, on_click=handle_ad_click, args=(ad_id,)):
                    pass

    # Display metrics
    st.header("Learning Progress")

    if len(st.session_state.clicks) > 0:
        # Calculate metrics
        total_interactions = len(st.session_state.clicks)
        avg_reward = sum(st.session_state.rewards) / len(st.session_state.rewards)
        
        # Display metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Interactions", f"{total_interactions}")
        col2.metric("Average Reward", f"{avg_reward:.4f}")
        
        # Plot category distribution
        if len(st.session_state.categories) > 0:
            st.subheader("Ad Categories Viewed")
            category_counts = pd.Series(st.session_state.categories).value_counts()
            st.bar_chart(category_counts)
            
        # Plot preference evolution
        if len(st.session_state.categories) > 5:
            st.subheader("Preference Evolution")
            fig, ax = plt.subplots()
            
            # Get current preferences
            prefs = st.session_state.simulator.user_preferences[user_id]
            categories = list(prefs.keys())
            values = list(prefs.values())
            
            # Create bar chart
            ax.bar(categories, values)
            ax.set_xlabel("Category")
            ax.set_ylabel("Preference Level")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        # Plot reward by category
        if user_id in st.session_state.category_avg_rewards and len(st.session_state.category_avg_rewards[user_id]) > 0:
            st.subheader("Average Reward by Category")
            fig, ax = plt.subplots()
            
            # Get average rewards by category
            cat_rewards = {}
            for cat, data in st.session_state.category_avg_rewards[user_id].items():
                if data['count'] > 0:
                    cat_rewards[cat] = data['total'] / data['count']
            
            if cat_rewards:
                categories = list(cat_rewards.keys())
                values = list(cat_rewards.values())
                
                # Create bar chart
                ax.bar(categories, values)
                ax.set_xlabel("Category")
                ax.set_ylabel("Average Reward")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
        # Plot learning curve
        if len(st.session_state.rewards) > 5:
            st.subheader("Learning Curve")
            fig, ax = plt.subplots()
            
            # Calculate moving average
            window_size = min(10, len(st.session_state.rewards))
            rewards_series = pd.Series(st.session_state.rewards)
            moving_avg = rewards_series.rolling(window=window_size).mean()
            
            ax.plot(moving_avg)
            ax.set_xlabel("Interaction")
            ax.set_ylabel("Reward (Moving Average)")
            st.pyplot(fig)
    else:
        st.info("Click on ads to see learning progress.")

    # Display user history
    st.header("User Interaction History")
    if user_history:
        history_df = pd.DataFrame(user_history[-10:])  # Show last 10 interactions
        
        # Select columns to display
        columns_to_display = ['timestamp', 'category', 'product', 'action', 'reward', 'ad_id']
        display_columns = [col for col in columns_to_display if col in history_df.columns]
        
        if display_columns:
            st.dataframe(history_df[display_columns])
        else:
            st.dataframe(history_df)
    else:
        st.info("No interaction history yet.")

elif st.session_state.current_page == "ad_detail":
    # Ad detail page
    ad_id = st.session_state.current_ad

    ad = get_ad_by_id(ad_id)

    if ad is None:
        st.error(f"Ad not found! ID: {ad_id}")
        if st.button("Return to Home"):
            go_back_to_home()
    else:
        # Back button
        if st.button("← Back to Recommendations"):
            go_back_to_home()
        
        # Display ad details
        st.header(ad['title'])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display ad image (placeholder)
            st.image(f"https://via.placeholder.com/300x200?text={ad['product']}", 
                    caption=f"{ad['product']} Image")
        
        with col2:
            st.subheader("Product Details")
            st.markdown(f"**Category:** {ad['category']}")
            st.markdown(f"**Product:** {ad['product']}")
            st.markdown(f"**Description:** {ad.get('description', 'No description available')}")
            
            # Add some fictional details
            st.markdown("### Features")
            st.markdown("- High quality product")
            st.markdown("- Best in class performance")
            st.markdown("- Excellent customer reviews")
            
            st.markdown("### Price")
            # Use ad_id as seed for consistent pricing
            random.seed(ad_id)
            price = random.randint(50, 500)
            st.markdown(f"**${price}.99**")
            
            # Add a buy button
            if st.button("Buy Now"):
                st.success("Purchase successful! (This is a simulation)")
                
                # Use the handle_purchase function for purchase
                handle_purchase(ad_id)
        
        # Related products section
        st.header("Related Products")
        
        # Get products from the same category
        same_category_ads = ads_df[ads_df['category'] == ad['category']]
        same_category_ads = same_category_ads[same_category_ads['ad_id'] != ad_id]
        
        # Ensure we have at least some products to show
        if len(same_category_ads) > 0:
            # Sample up to 3 related products
            sample_size = min(3, len(same_category_ads))
            same_category_ads = same_category_ads.sample(sample_size)
            
            cols = st.columns(sample_size)
            for i, (_, related_ad) in enumerate(same_category_ads.iterrows()):
                related_ad_id = related_ad['ad_id']
                with cols[i]:
                    st.subheader(related_ad['title'])
                    st.text(f"Product: {related_ad['product']}")
                    
                    # Sử dụng callback function với related_ad_id cụ thể
                    button_key = get_unique_key("related", related_ad_id)
                    if st.button(f"View Product", key=button_key, on_click=handle_ad_click, args=(related_ad_id,)):
                        pass
        else:
            st.info("No related products found.")

# Instructions
st.sidebar.markdown("""
## How to use
1. Select a user from the dropdown
2. Click on ads that interest you
3. The system will learn your preferences
4. Watch how recommendations improve over time
""")