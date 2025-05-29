import numpy as np
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay, alpha=1, beta=2):
        """
        初始化蚁群算法参数
        distances: 距离矩阵
        n_ants: 蚂蚁数量
        n_iterations: 迭代次数
        decay: 信息素衰减率
        alpha: 信息素重要性参数
        beta: 启发式信息重要性参数
        """
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        # 初始化信息素矩阵，初始值设为较小的常数
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        
        # 防止除以零错误
        self.distances[self.distances == 0] = 1e-10
        
        # 启发式信息矩阵（距离的倒数）
        self.heuristic = 1.0 / self.distances
        
    def run(self):
        """执行蚁群算法"""
        best_path = None
        best_distance = float('inf')
        
        for iteration in range(self.n_iterations):
            all_paths = self._generate_all_paths()
            self._update_pheromone(all_paths)
            
            # 找出本次迭代的最优路径
            current_best_path, current_best_distance = self._get_best_path(all_paths)
            
            # 更新全局最优路径
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_path = current_best_path
                
            # 打印进度
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 最优路径长度 = {best_distance:.2f}")
                
        return best_path, best_distance
    
    def _generate_all_paths(self):
        """生成所有蚂蚁的路径"""
        all_paths = []
        for ant in range(self.n_ants):
            path = self._generate_path()
            all_paths.append(path)
        return all_paths
    
    def _generate_path(self):
        """生成一只蚂蚁的路径"""
        path = []
        visited = set()
        
        # 随机选择起点
        start_node = np.random.randint(0, len(self.distances))
        path.append(start_node)
        visited.add(start_node)
        
        # 构建路径
        for _ in range(len(self.distances) - 1):
            current_node = path[-1]
            # 选择下一个节点
            next_node = self._select_next_node(current_node, visited)
            path.append(next_node)
            visited.add(next_node)
            
        # 回到起点形成回路
        path.append(path[0])
        
        return path
    
    def _select_next_node(self, current_node, visited):
        """选择下一个节点"""
        unvisited = set(range(len(self.distances))) - visited
        probabilities = np.zeros(len(self.distances))
        
        for node in unvisited:
            # 计算转移概率
            probabilities[node] = (self.pheromone[current_node, node] ** self.alpha) * \
                                 (self.heuristic[current_node, node] ** self.beta)
        
        # 归一化概率
        if sum(probabilities) > 0:
            probabilities /= sum(probabilities)
        else:
            # 如果所有概率都为0，随机选择一个未访问的节点
            return list(unvisited)[np.random.randint(0, len(unvisited))]
        
        # 根据概率选择下一个节点
        next_node = np.random.choice(range(len(self.distances)), p=probabilities)
        
        return next_node
    
    def _update_pheromone(self, all_paths):
        """更新信息素矩阵"""
        # 信息素挥发
        self.pheromone *= (1 - self.decay)
        
        # 每只蚂蚁更新信息素
        for path in all_paths:
            path_distance = self._calculate_path_distance(path)
            
            for i in range(len(path) - 1):
                node_i = path[i]
                node_j = path[i + 1]
                # 信息素增量与路径长度成反比
                self.pheromone[node_i, node_j] += 1.0 / path_distance
    
    def _calculate_path_distance(self, path):
        """计算路径总长度"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distances[path[i], path[i + 1]]
        return total_distance
    
    def _get_best_path(self, all_paths):
        """找出所有路径中的最优路径"""
        best_distance = float('inf')
        best_path = None
        
        for path in all_paths:
            distance = self._calculate_path_distance(path)
            if distance < best_distance:
                best_distance = distance
                best_path = path
                
        return best_path, best_distance

# 示例使用
if __name__ == "__main__":
    # 创建一个简单的距离矩阵（10个城市）
    n_cities = 10
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    cities = np.random.rand(n_cities, 2)  # 随机生成城市坐标
    
    # 计算距离矩阵
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distances[i, j] = np.sqrt(((cities[i] - cities[j]) ** 2).sum())
    
    # 初始化蚁群算法
    ant_colony = AntColony(
        distances=distances,
        n_ants=20,
        n_iterations=100,
        decay=0.5,
        alpha=1,
        beta=2
    )
    
    # 运行算法
    best_path, best_distance = ant_colony.run()
    
    print(f"\n最优路径: {best_path}")
    print(f"最优路径长度: {best_distance:.2f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 8))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=100)
    
    # 绘制最优路径
    for i in range(len(best_path) - 1):
        plt.plot(
            [cities[best_path[i], 0], cities[best_path[i+1], 0]],
            [cities[best_path[i], 1], cities[best_path[i+1], 1]],
            'r-', linewidth=2
        )
    
    plt.title('蚁群算法解决TSP问题')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)
    plt.show()
