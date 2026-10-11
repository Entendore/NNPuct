"""
NNEvolutionEngine: Topological Neural Network Optimization via MCTS (v3 - Stable)

Final refinements include:
1. Layer Normalization on PUCT input to handle magnitude mismatches between weights and 4th-order derivatives.
2. Glorot/Xavier initialization for TargetNN to ensure healthy tanh gradients.
3. Experience Replay Buffer for stable PUCT predictor training across generations.
4. Standard MCTS lazy evaluation and correct backpropagation.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import List, Tuple, Optional
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class TargetNN:
    """
    The Optimization State. Uses tanh activations and Glorot initialization 
    to ensure a smooth, healthy loss landscape with meaningful higher-order derivatives.
    """
    def __init__(self, architecture: List[int], weights: Optional[List[np.ndarray]] = None, 
                 biases: Optional[List[np.ndarray]] = None):
        self.architecture = architecture
        
        if weights is None:
            # Glorot Uniform is mathematically optimal for tanh networks
            init = tf.keras.initializers.GlorotUniform()
            self.weights = [tf.Variable(init([architecture[i], architecture[i+1]]))
                            for i in range(len(architecture) - 1)]
            self.biases = [tf.Variable(tf.zeros([architecture[i+1]]))
                           for i in range(len(architecture) - 1)]
        else:
            self.weights = [tf.Variable(w) for w in weights]
            self.biases = [tf.Variable(b) for b in biases]

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b
            if i < len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return h

    def get_topological_hash(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[np.ndarray, float, float, List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        eps = 1e-5
        
        with tf.GradientTape(persistent=True) as tape4:
            with tf.GradientTape(persistent=True) as tape3:
                with tf.GradientTape(persistent=True) as tape2:
                    with tf.GradientTape() as tape1:
                        logits = self.forward(x)
                        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
                    
                    grads_w = tape1.gradient(loss, self.weights) 
                    grads_w = [g if g is not None else tf.zeros_like(w) for g, w in zip(grads_w, self.weights)]
                
                hessians_w = [tape2.gradient(g, w) if g is not None else tf.zeros_like(w) 
                              for g, w in zip(grads_w, self.weights)]
                hessians_w = [h if h is not None else tf.zeros_like(w) for h, w in zip(hessians_w, self.weights)]
            
                grads_b = tape2.gradient(loss, self.biases)
                grads_b = [g if g is not None else tf.zeros_like(b) for g, b in zip(grads_b, self.biases)]
            
            third_derivs_w = [tape3.gradient(h, w) if h is not None else tf.zeros_like(w) 
                              for h, w in zip(hessians_w, self.weights)]
            third_derivs_w = [t if t is not None else tf.zeros_like(w) for t, w in zip(third_derivs_w, self.weights)]
        
        fourth_derivs_w = [tape4.gradient(t, w) if t is not None else tf.zeros_like(w) 
                           for t, w in zip(third_derivs_w, self.weights)]
        fourth_derivs_w = [f if f is not None else tf.zeros_like(w) for f, w in zip(fourth_derivs_w, self.weights)]

        del tape1, tape2, tape3, tape4

        state_hash = []
        state_hash.append(loss)
        
        softmax_probs = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(softmax_probs * tf.math.log(softmax_probs + eps), axis=1)
        state_hash.append(tf.reduce_mean(entropy))
        
        layer_norms_g = [] 
        
        for w, g, h, t, f in zip(self.weights, grads_w, hessians_w, third_derivs_w, fourth_derivs_w):
            tensors = [w, g, h, t, f]
            layer_norms_g.append(tf.norm(g))
            
            for tensor in tensors:
                flat_t = tf.reshape(tensor, [-1])
                mean_val = tf.reduce_mean(flat_t)
                l2_norm = tf.norm(flat_t)
                std_val = tf.math.reduce_std(flat_t)
                max_abs = tf.reduce_max(tf.abs(flat_t))
                sparsity = tf.reduce_mean(tf.cast(tf.abs(flat_t) < eps, tf.float32))
                skewness = tf.reduce_mean(((flat_t - mean_val) / (std_val + eps)) ** 3)
                kurtosis = tf.reduce_mean(((flat_t - mean_val) / (std_val + eps)) ** 4) - 3.0
                state_hash.extend([mean_val, l2_norm, std_val, max_abs, sparsity, skewness, kurtosis])
        
        max_g = tf.reduce_max(layer_norms_g)
        min_g = tf.reduce_min(layer_norms_g)
        state_hash.append(tf.math.log(max_g / (min_g + eps) + eps))
        
        state_hash_np = tf.stack(state_hash).numpy().astype(np.float32)
        acc = tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1) == y, tf.float32))
        
        return state_hash_np, acc.numpy(), loss.numpy(), grads_w, grads_b, hessians_w, third_derivs_w, fourth_derivs_w

    def apply_move(self, action_idx: int, grads_w, grads_b, hessians, third_derivs, fourth_derivs, alpha: float = 0.05) -> 'TargetNN':
        layer_idx = action_idx // 8
        operation_idx = action_idx % 8
        
        new_weights = [w.numpy().copy() for w in self.weights]
        new_biases = [b.numpy().copy() for b in self.biases]
        
        g = grads_w[layer_idx].numpy()
        h = hessians[layer_idx].numpy()
        t = third_derivs[layer_idx].numpy()
        f = fourth_derivs[layer_idx].numpy()
        g_b = grads_b[layer_idx].numpy()
        
        if operation_idx == 0: 
            update_w = alpha * g
        elif operation_idx == 1: 
            update_w = alpha * (g / (1.0 + np.abs(h)))
        elif operation_idx == 2: 
            saddle_mask = (h < 0).astype(np.float32)
            update_w = alpha * g * (1.0 + saddle_mask) 
        elif operation_idx == 3: 
            update_w = alpha * g / (1.0 + np.abs(t))
        elif operation_idx == 4: 
            update_w = -alpha * 0.01 * np.sign(t)
        elif operation_idx == 5: 
            update_w = alpha * g / (1.0 + np.abs(f))
        elif operation_idx == 6: 
            noise = np.random.normal(0.0, 1.0, size=g.shape).astype(np.float32)
            update_w = alpha * g + alpha * 0.01 * np.abs(f) * noise
        elif operation_idx == 7: 
            noise = np.random.normal(0.0, 1.0, size=g.shape).astype(np.float32)
            newton = g / (1.0 + np.abs(h))
            jerk_damp = 1.0 / (1.0 + 0.1 * np.abs(t))
            snap_noise = 0.001 * np.abs(f) * noise
            update_w = alpha * (newton * jerk_damp + snap_noise)
            
        update_b = alpha * g_b
        
        new_weights[layer_idx] = np.clip(new_weights[layer_idx] - update_w, -50.0, 50.0)
        new_biases[layer_idx] = np.clip(new_biases[layer_idx] - update_b, -50.0, 50.0)
        
        return TargetNN(self.architecture, weights=new_weights, biases=new_biases)


class PUCTPredictor(tf.keras.Model):
    """Search heuristic with Layer Normalization to handle scale mismatches."""
    def __init__(self, hash_size: int, num_actions: int):
        super().__init__()
        # Normalization is critical: weights (~5.0) and 4th derivatives (~0.00001) 
        # must be evaluated on equal footing.
        self.input_norm = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.policy_head = tf.keras.layers.Dense(num_actions, activation='softmax')
        self.value_head = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, state_hash: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.input_norm(state_hash)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.policy_head(x), self.value_head(x)


class MCTSNode:
    def __init__(self, nn_state: TargetNN, state_hash: Optional[np.ndarray], reward: float, 
                 parent: Optional['MCTSNode'] = None, action: Optional[int] = None):
        self.nn = nn_state              
        self.hash = state_hash         
        self.reward = reward           
        self.parent = parent
        self.action = action           
        self.children: List['MCTSNode'] = []
        self.N = 0                     
        self.W = 0                     
        self.Q = 0                     
        self.P = 0                     
        self.is_evaluated = False      
        self.g_w, self.g_b, self.h, self.t, self.f = None, None, None, None, None


class MCTS_PUCT:
    def __init__(self, predictor: PUCTPredictor, num_actions: int, c_puct: float = 1.5):
        self.predictor = predictor
        self.num_actions = num_actions
        self.c_puct = c_puct

    def search(self, root_nn: TargetNN, root_hash: np.ndarray, root_reward: float, 
               root_g_w, root_g_b, root_h, root_t, root_f, x_data: tf.Tensor, y_data: tf.Tensor, 
               simulations: int = 15) -> Tuple[MCTSNode, np.ndarray, float]:
        
        root = MCTSNode(root_nn, root_hash, root_reward)
        root.g_w, root.g_b, root.h, root.t, root.f = root_g_w, root_g_b, root_h, root_t, root_f
        root.is_evaluated = True
        
        p, _ = self.predictor(tf.expand_dims(root_hash, 0))
        noise = np.random.dirichlet([0.3] * self.num_actions)
        root.P = 0.75 * p.numpy()[0] + 0.25 * noise

        for a in range(self.num_actions):
            child_nn = root.nn.apply_move(a, root.g_w, root.g_b, root.h, root.t, root.f)
            child = MCTSNode(child_nn, None, 0.0, parent=root, action=a)
            root.children.append(child)

        for _ in range(simulations):
            node = root
            
            while node.children and node.is_evaluated:
                node = self.select_child(node)
                
            if not node.is_evaluated:
                child_hash, child_reward, _, g_w, g_b, h, t, f = node.nn.get_topological_hash(x_data, y_data)
                node.hash = child_hash
                node.reward = child_reward
                node.g_w, node.g_b, node.h, node.t, node.f = g_w, g_b, h, t, f
                node.is_evaluated = True
                
                cp, cv = self.predictor(tf.expand_dims(child_hash, 0))
                node.P = cp.numpy()[0][node.action]
                node.Q = cv.numpy()[0][0]
                
                for a in range(self.num_actions):
                    child_nn = node.nn.apply_move(a, node.g_w, node.g_b, node.h, node.t, node.f)
                    grandchild = MCTSNode(child_nn, None, 0.0, parent=node, action=a)
                    node.children.append(grandchild)
            
            self.backpropagate(node, node.reward)
            
        visit_counts = np.array([c.N for c in root.children], dtype=np.float32)
        sum_visits = np.sum(visit_counts)
        policy_target = visit_counts / sum_visits if sum_visits > 0 else np.ones_like(visit_counts) / len(visit_counts)
        
        evaluated_rewards = [c.reward for c in root.children if c.is_evaluated]
        value_target = max(evaluated_rewards) if evaluated_rewards else root.reward
        
        return root, policy_target, value_target

    def select_child(self, node: MCTSNode) -> MCTSNode:
        best_child, best_puct = None, -float('inf')
        total_N = sum(c.N for c in node.children)
        
        for child in node.children:
            U = self.c_puct * child.P * (np.sqrt(total_N + 1) / (1 + child.N))
            puct = child.Q + U
            if puct > best_puct:
                best_puct, best_child = puct, child
        return best_child

    def backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.N += 1
            node.W += reward
            node.Q = node.W / node.N
            node = node.parent


class NNEvolutionEngine:
    def __init__(self, architecture: List[int], sample_size: int, generations: int):
        self.architecture = architecture
        self.sample_size = sample_size
        self.generations = generations
        
        self.num_layers = len(architecture) - 1
        self.num_actions = self.num_layers * 8
        
        self.features_per_layer = 35            
        self.hash_size = (self.num_layers * self.features_per_layer) + 3 
        
        self.predictor = PUCTPredictor(self.hash_size, self.num_actions)
        self.mcts = MCTS_PUCT(self.predictor, self.num_actions)
        self.predictor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Experience Replay Buffer for stable PUCT training
        self.replay_buffer = deque(maxlen=200)

    def train_puct_predictor(self):
        if len(self.replay_buffer) < 8:
            return 0.0
            
        # Sample a mini-batch from replay buffer
        batch_indices = np.random.choice(len(self.replay_buffer), size=8, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        state_hashes = np.array([b[0] for b in batch])
        policy_targets = np.array([b[1] for b in batch])
        value_targets = np.array([b[2] for b in batch])

        with tf.GradientTape() as tape:
            pred_policy, pred_value = self.predictor(state_hashes)
            
            policy_loss = tf.keras.losses.categorical_crossentropy(policy_targets, pred_policy)
            value_loss = tf.keras.losses.mse(value_targets, tf.squeeze(pred_value))
            total_loss = tf.reduce_mean(policy_loss + value_loss)
            
        grads = tape.gradient(total_loss, self.predictor.trainable_variables)
        self.predictor_optimizer.apply_gradients(zip(grads, self.predictor.trainable_variables))
        return total_loss.numpy()

    def evolve(self, x_data: tf.Tensor, y_data: tf.Tensor):
        logger.info("Initializing Generation 1 with Glorot-initialized NNs...")
        population = [TargetNN(self.architecture) for _ in range(self.sample_size)]
        best_overall_acc = 0.0
        best_overall_nn = None
        
        for gen in range(self.generations):
            logger.info(f"--- Generation {gen+1}/{self.generations} ---")
            new_population = []
            
            for i, nn in enumerate(population):
                try:
                    h, acc, loss, g_w, g_b, hess, t, f = nn.get_topological_hash(x_data, y_data)
                    
                    if acc > best_overall_acc:
                        best_overall_acc = acc
                        best_overall_nn = nn
                        logger.info(f"🎉 New Best Accuracy Found: {best_overall_acc:.4f}")
                    
                    root, policy_target, value_target = self.mcts.search(
                        nn, h, acc, g_w, g_b, hess, t, f, x_data, y_data, simulations=15
                    )
                    
                    # Save to replay buffer instead of immediate fitting
                    self.replay_buffer.append((h, policy_target, value_target))
                    
                    # Train PUCT on historical data
                    puct_loss = self.train_puct_predictor()
                    logger.info(f"Gen {gen+1} NN {i} -> Acc: {acc:.4f} | PUCT Loss: {puct_loss:.4f}")
                    
                    if root.children:
                        best_child = max(root.children, key=lambda c: c.Q)
                        new_population.append(best_child.nn)
                    else:
                        new_population.append(nn)
                        
                except Exception as e:
                    logger.error(f"Error processing NN {i}: {e}. Keeping original parent.")
                    new_population.append(nn) 
                    
            population = new_population
            
        logger.info(f"Evolution Complete. Best Benchmark Accuracy: {best_overall_acc:.4f}")
        return best_overall_nn if best_overall_nn else population[0]


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
            
    x_data = tf.constant([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=tf.float32)
    y_data = tf.constant([0, 1, 1, 0], dtype=tf.int32)
    
    architecture = [2, 8, 2] 
    sample_size = 5           
    generations = 10          
    
    engine = NNEvolutionEngine(architecture, sample_size, generations)
    best_model = engine.evolve(x_data, y_data)