# viper_graph.py

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def vertical_hierarchy_pos(G, root="Start", xcenter=0.5, width=1., vert_gap=0.25, vert_loc=0):
    pos = {}
    def _recurse(node, x, y, width):
        pos[node] = (x, y)
        children = list(G.neighbors(node))
        if not children:
            return
        dx = width / len(children)
        nextx = x - width/2 + dx/2
        for child in children:
            _recurse(child, nextx, y - vert_gap, dx)
            nextx += dx
    _recurse(root, xcenter, vert_loc, width)
    return pos

def draw_viper_tree(q_table, ad_database, user_id=None, state=None, top_k=5, q_threshold=0.01):

    """
    Vẽ cây: Start → Các state có Q-value > threshold → Top-k quảng cáo (kèm category).
    Không vẽ những state chưa học được (Q quá thấp).
    """

    G = nx.DiGraph()
    G.add_node("Start")

    num_states = q_table.shape[0]

    for state in range(num_states):
        max_q = np.max(q_table[state])
        if max_q < q_threshold:
            continue  # bỏ qua state chưa học được gì

        state_label = f"State {state}"
        G.add_edge("Start", state_label)

        top_ads = q_table[state].argsort()[-top_k:][::-1]
        for ad_idx in top_ads:
            ad = ad_database[ad_database['ad_id'] == ad_idx]
            if not ad.empty:
                ad = ad.iloc[0]
                label = f"Ad {ad['ad_id']}: {ad['product']} ({ad['category']})"
            else:
                label = f"Ad {ad_idx}"
            G.add_edge(state_label, label)

    pos_vertical = vertical_hierarchy_pos(G, vert_gap=0.3, vert_loc=1.0)
    pos_horizontal = {node: (y, -x) for node, (x, y) in pos_vertical.items()}

    plt.figure(figsize=(22, 10))
    nx.draw(
        G, pos_horizontal,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        node_size=2500,
        font_size=8,
        font_weight='bold'
    )
    title = f"VIPER Tree (User {user_id}) — Only States with Learned Q-values"
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


