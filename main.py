"""
NNEvolutionEngine: Topological State-Space Optimization for Emergent Intelligence

This engine transcends standard optimization by treating weight configuration as a 
dynamical system. Instead of minimizing training loss, the MCTS optimizes for 
Generalization (Validation Accuracy) and Latent Space Health (Activation Variance).

By measuring the topological features of the loss landscape up to the 4th derivative, 
and combining them with the structural topology of the hidden activations, the search 
heuristics learn to cultivate networks that compute features rather than memorize data.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import List, Tuple, Optional
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tf.random.set_seed(42)
np.random.seed(42)


class TargetState:
    """
    Represents a point in the neural network's state-space.
    Uses tanh activations and Glorot initialization. 
    Instruments hidden layers to measure the health of the emergent latent space.
    """
    def __init__(self, architecture: List[int], weights: Optional[List[np.ndarray]] = None, 
                 biases: Optional[List[np.ndarray]] = None):
        self.architecture = architecture
        
        if weights is None:
            init = tf.keras.initializers.GlorotUniform()
            self.weights = [tf.Variable(init([architecture[i], architecture[i+1]]))
                            for i in range(len(architecture) - 1)]
            self.biases = [tf.Variable(tf.zeros([architecture[i+1]]))
                           for i in range(len(architecture) - 1)]
        else:
            self.weights = [tf.Variable(w) for w in weights]
            self.biases = [tf.Variable(b) for b in biases]

    def forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Returns logits and the hidden layer activations (the latent space)."""
        h = x
        activations = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b
            if i < len(self.weights) - 1:
                h = tf.nn.tanh(h)
                activations.append(h) # Store latent representations
        return h, activations

    def get_topological_hash(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[np.ndarray, float, float, List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        eps = 1e-5
        
        with tf.GradientTape(persistent=True) as tape4:
            with tf.GradientTape(persistent=True) as tape3:
                with tf.GradientTape(persistent=True) as tape2:
                    with tf.GradientTape() as tape1:
                        logits, activations = self.forward(x)
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

        state_hash = [loss]
        
        softmax_probs = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(softmax_probs * tf.math.log(softmax_probs + eps), axis=1)
        state_hash.append(tf.reduce_mean(entropy))
        
        # --- LATENT SPACE HEALTH METRIC ---
        # Activation Variance: Measures if the network is actually using the hidden layer
        # to compute distinct features for distinct inputs, rather than collapsing to a constant.
        latent_variance = tf.reduce_mean(tf.math.reduce_variance(activations[0], axis=0))
        state_hash.append(latent_variance)
        
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

    def apply_transition(self, action_idx: int, grads_w, grads_b, hessians, third_derivs, fourth_derivs, alpha: float = 0.05) -> 'TargetState':
        """Executes a state transition operator on the weight matrices."""
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
        
        return TargetState(self.architecture, weights=new_weights, biases=new_biases)


class HeuristicPredictor(tf.keras.Model):
    """Search heuristic with Layer Normalization to handle scale mismatches."""
    def __init__(self, hash_size: int, num_actions: int):
        super().__init__()
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


class StateNode:
    """Standard lazy MCTS Node for state-space search."""
    def __init__(self, parent: Optional['StateNode'] = None, action: Optional[int] = None):
        self.parent = parent
        self.action = action           
        self.children: List['StateNode'] = []
        self.N = 0                     
        self.W = 0                     
        self.Q = 0                     
        self.P = 0                     
        
        self.is_evaluated = False      
        self.state: Optional[TargetState] = None
        self.hash: Optional[np.ndarray] = None
        self.reward: float = 0.0
        self.g_w, self.g_b, self.h, self.t, self.f = None, None, None, None, None

    def expand(self, policy_probs: np.ndarray):
        for a, p in enumerate(policy_probs):
            child = StateNode(parent=self, action=a)
            child.P = p
            self.children.append(child)


class MCTS_PUCT:
    def __init__(self, predictor: HeuristicPredictor, num_actions: int, c_puct: float = 1.5):
        self.predictor = predictor
        self.num_actions = num_actions
        self.c_puct = c_puct

    def search(self, root_state: TargetState, root_hash: np.ndarray, root_reward: float, 
               root_g_w, root_g_b, root_h, root_t, root_f, 
               x_train: tf.Tensor, y_train: tf.Tensor, x_val: tf.Tensor, y_val: tf.Tensor, 
               simulations: int = 20) -> Tuple[StateNode, np.ndarray, float]:
        
        root = StateNode()
        root.state = root_state
        root.hash = root_hash
        root.reward = root_reward
        root.g_w, root.g_b, root.h, root.t, root.f = root_g_w, root_g_b, root_h, root_t, root_f
        root.is_evaluated = True
        
        p, _ = self.predictor(tf.expand_dims(root_hash, 0))
        p = p.numpy()[0]
        noise = np.random.dirichlet([0.3] * self.num_actions)
        p = 0.75 * p + 0.25 * noise
        root.expand(p)

        for _ in range(simulations):
            node = root
            
            # 1. Selection
            while node.children:
                node = self.select_child(node)
                
            # 2. Expansion & Evaluation (Lazy)
            parent = node.parent
            node.state = parent.state.apply_transition(node.action, parent.g_w, parent.g_b, parent.h, parent.t, parent.f)
            
            # Evaluate on TRAIN data to get topological gradients
            child_hash, train_acc, _, g_w, g_b, h, t, f = node.state.get_topological_hash(x_train, y_train)
            node.hash = child_hash
            node.g_w, node.g_b, node.h, node.t, node.f = g_w, g_b, h, t, f
            node.is_evaluated = True
            
            # Evaluate on VALIDATION data to get true generalization reward
            val_logits, _ = node.state.forward(x_val)
            val_acc = tf.reduce_mean(tf.cast(tf.argmax(val_logits, axis=1) == y_val, tf.float32)).numpy()
            
            # Extract Latent Health from Hash (Index 2 is latent_variance)
            latent_health = child_hash[2] 
            
            # REWARD: Generalization + Latent Space Health
            # This forces the search to value networks that build useful representations
            # rather than just hardcoding the training set answers.
            node.reward = val_acc + (latent_health * 0.1)
            
            p_c, _ = self.predictor(tf.expand_dims(child_hash, 0))
            node.expand(p_c.numpy()[0])
            
            # 3. Backpropagation
            self.backpropagate(node, node.reward)
            
        # 4. Training Targets
        visit_counts = np.array([c.N for c in root.children], dtype=np.float32)
        sum_visits = np.sum(visit_counts)
        policy_target = visit_counts / sum_visits if sum_visits > 0 else np.ones_like(visit_counts) / len(visit_counts)
        
        evaluated_rewards = [c.reward for c in root.children if c.is_evaluated]
        value_target = max(evaluated_rewards) if evaluated_rewards else root.reward
        
        return root, policy_target, value_target

    def select_child(self, node: StateNode) -> StateNode:
        best_child, best_puct = None, -float('inf')
        total_N = node.N 
        
        for child in node.children:
            U = self.c_puct * child.P * (np.sqrt(total_N + 1) / (1 + child.N))
            puct = child.Q + U
            if puct > best_puct:
                best_puct, best_child = puct, child
        return best_child

    def backpropagate(self, node: StateNode, reward: float):
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
        # +4 for loss, entropy, latent_variance, and gradient ratio
        self.hash_size = (self.num_layers * self.features_per_layer) + 4 
        
        self.predictor = HeuristicPredictor(self.hash_size, self.num_actions)
        self.mcts = MCTS_PUCT(self.predictor, self.num_actions)
        self.predictor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.replay_buffer = deque(maxlen=256)

    def train_puct_predictor(self):
        if len(self.replay_buffer) < 16:
            return 0.0
            
        batch_indices = np.random.choice(len(self.replay_buffer), size=16, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        state_hashes = np.array([b[0] for b in batch])
        policy_targets = np.array([b[1] for b in batch])
        value_targets = np.array([b[2] for b in batch])

        with tf.GradientTape() as tape:
            pred_policy, pred_value = self.predictor(state_hashes)
            policy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(policy_targets, pred_policy))
            value_loss = tf.reduce_mean(tf.keras.losses.mse(value_targets, tf.squeeze(pred_value)))
            total_loss = policy_loss + value_loss
            
        grads = tape.gradient(total_loss, self.predictor.trainable_variables)
        self.predictor_optimizer.apply_gradients(zip(grads, self.predictor.trainable_variables))
        return total_loss.numpy()

    def evolve(self, x_train: tf.Tensor, y_train: tf.Tensor, x_val: tf.Tensor, y_val: tf.Tensor):
        logger.info("Initializing Generation 1 with Glorot-initialized states...")
        population = [TargetState(self.architecture) for _ in range(self.sample_size)]
        best_overall_reward = -1.0
        best_overall_state = None
        
        for gen in range(self.generations):
            logger.info(f"--- Generation {gen+1}/{self.generations} ---")
            new_population = []
            
            for i, state in enumerate(population):
                try:
                    h, train_acc, loss, g_w, g_b, hess, t, f = state.get_topological_hash(x_train, y_train)
                    
                    # Evaluate Initial Generalization
                    val_logits, _ = state.forward(x_val)
                    val_acc = tf.reduce_mean(tf.cast(tf.argmax(val_logits, axis=1) == y_val, tf.float32)).numpy()
                    current_reward = val_acc + (h[2] * 0.1) # val_acc + latent_health
                    
                    if current_reward > best_overall_reward:
                        best_overall_reward = current_reward
                        best_overall_state = state
                        logger.info(f"🎉 New Best Reward (Val Acc: {val_acc:.4f}) Found!")

                    if val_acc >= 1.0 and train_acc >= 1.0:
                        logger.info("Perfect Generalization reached. Halting evolution.")
                        return best_overall_state
                    
                    root, policy_target, value_target = self.mcts.search(
                        state, h, current_reward, g_w, g_b, hess, t, f, 
                        x_train, y_train, x_val, y_val, simulations=20
                    )
                    
                    self.replay_buffer.append((h, policy_target, value_target))
                    puct_loss = self.train_puct_predictor()
                    
                    action_names = ["GradDesc", "Newton", "SaddleEsc", "JerkDamp", "JerkDrift", "SnapDamp", "SnapNoise", "MixedTopo"]
                    top_action_idx = np.argmax(policy_target)
                    op_idx = top_action_idx % 8
                    layer_idx = top_action_idx // 8
                    logger.info(f"Gen {gen+1} State {i} -> Train: {train_acc:.2f} | Val: {val_acc:.2f} | PUCT Loss: {puct_loss:.4f} | Top Op: {action_names[op_idx]} on L{layer_idx}")
                    
                    # Select best child based on Mean Value (Q) which encapsulates generalization
                    if root.children:
                        best_child = max(root.children, key=lambda c: c.Q)
                        new_population.append(best_child.state)
                    else:
                        new_population.append(state)
                        
                except Exception as e:
                    logger.error(f"Error processing State {i}: {e}. Keeping original parent.")
                    new_population.append(state) 
                    
            population = new_population
            
        logger.info(f"Evolution Complete. Best Generalization Reward: {best_overall_reward:.4f}")
        return best_overall_state if best_overall_state else population[0]


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
            
    # --- 3-BIT PARITY BENCHMARK ---
    # A network cannot simply memorize this; it must learn the concept of parity (XOR logic).
    # 8 possible inputs. We train on 6, and test on 2 unseen states.
    
    # Generate all 8 combinations of 3 bits
    all_inputs = np.array([[int(b) for b in format(i, '03b')] for i in range(8)], dtype=np.float32)
    # Parity is 1 if odd number of 1s, 0 if even
    all_outputs = np.array([np.sum(bits) % 2 for bits in all_inputs], dtype=np.int32)
    
    # Hold out 2 states for validation (e.g., 101 and 010)
    train_mask = np.array([True, True, True, True, False, True, False, True])
    
    x_train = tf.constant(all_inputs[train_mask])
    y_train = tf.constant(all_outputs[train_mask])
    
    x_val = tf.constant(all_inputs[~train_mask])
    y_val = tf.constant(all_outputs[~train_mask])
    
    # Architecture requires enough capacity to form a latent space for parity
    architecture = [3, 16, 2] 
    sample_size = 6           
    generations = 20          
    
    engine = NNEvolutionEngine(architecture, sample_size, generations)
    best_model = engine.evolve(x_train, y_train, x_val, y_val)