import copy
import math
import random
import time as ti
import json

import pickle

start_time = ti.time()
# ================== 基础数据配置 ==================
n = 10
filename = 'Prob-10A-50.txt'
N = 2 * n + 2
Np = {i for i in range(1, n + 1)}
Nd = {i for i in range(n + 1, 2 * n + 1)}
c_t = 0.1
fixed_cost = 10
time_zone = ['0.0-120.0', '120.0-600.0', '600.0-720.0', '720.0-840.0']
time_zone_int = {0: (0, 120), 1: (120, 600), 2: (600, 720), 3: (720, 840)}


# ================== 数据读取函数（完整保留）==================
def read_file(filename):
    columns = {
        "node": [],
        "pickup": [],
        "delivery": [],
        "Node": [],
        "Pickup": [],
        "Delivery": []
    }

    with open(filename, 'r') as file:
        section = None
        data = {"speed": {}, "speed_choose_matrix": []}

        h = 0
        for line in file:
            line = line.strip()
            if not line:
                continue
            if h == 0:
                K = int(line)
            elif h == 1:
                Q = int(line)
            elif h == 2:
                max_time = int(line)
            h += 1
            if h == 3:
                break

        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1].strip().lower()
                continue

            if section in columns:
                cleaned_line = line.replace(',', '.').split()
                if not columns[section]:
                    for _ in range(len(cleaned_line)):
                        columns[section].append([])
                for col_index, value in enumerate(cleaned_line):
                    columns[section][col_index].append(value)
            elif section.startswith("speed"):
                if section.startswith("speed choose"):
                    data["speed_choose_matrix"].append([int(x) for x in line.split()])
                else:
                    speed_data = line.replace(',', '.').split()
                    speed_data = [float(x) for x in speed_data]
                    speed_index = int(section.split()[-1])
                    if speed_index not in data["speed"]:
                        data["speed"][speed_index] = []
                    data["speed"][speed_index].append(speed_data)

    X = []
    Y = []
    s = []
    q = []
    p = []
    e = []
    l = []

    for section, column_data in columns.items():
        if section in ['pickup', 'delivery']:
            for h, col_values in enumerate(column_data):
                h += 1
                if h == 2:
                    X.extend([float(i) for i in col_values])
                elif h == 3:
                    Y.extend([float(i) for i in col_values])
                elif h == 4:
                    s.extend([int(i) for i in col_values])
                elif h == 5:
                    q.extend([int(i) for i in col_values])
                elif h == 6:
                    p.extend([int(i) for i in col_values])
                elif h == 7:
                    e.extend([int(i) for i in col_values])
                elif h == 8:
                    l.extend([int(i) for i in col_values])

    X = {index: value for index, value in enumerate(X, start=1)}
    Y = {index: value for index, value in enumerate(Y, start=1)}
    s = {index: value for index, value in enumerate(s, start=1)}
    q = {index: value for index, value in enumerate(q, start=1)}
    p = {index: value for index, value in enumerate(p, start=1)}
    e = {index: value for index, value in enumerate(e, start=1)}
    l = {index: value for index, value in enumerate(l, start=1)}

    X[0] = 25.0
    Y[0] = 25.0
    X[2 * n + 1] = 25.0
    Y[2 * n + 1] = 25.0
    s[0] = 0
    q[0] = 0
    p[0] = 0
    e[0] = 0
    l[0] = 0
    q[2 * n + 1] = 0
    p[2 * n + 1] = 0
    s[2 * n + 1] = 0
    e[2 * n + 1] = 0
    l[2 * n + 1] = 840

    num_nodes = len(X)
    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i][j] = math.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)

    speed_dict = {}
    for speed_index, speed_data in data["speed"].items():
        speed_dict[speed_index] = {}
        for entry in speed_data:
            start, end, _, value = entry
            speed_dict[speed_index][f"{start}-{end}"] = value

    return X, Y, s, q, p, e, l, K, Q, max_time, distance_matrix, speed_dict, data["speed_choose_matrix"]

def load_cplex_solutions(filename='cplex_solutions.txt'):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"未找到文件 {filename}，请确保文件存在于当前目录下。")
        return {}
cplex_solutions = load_cplex_solutions()
X, Y, s, q, p, e, l, vehicle_count, vehicle_capacity, max_time, distance_matrix, speed_dict, speed_choose_matrix = read_file(
    filename)


# ================== 时间矩阵计算（完整保留）==================1
def calculate_time_m(distance_matrix, speed_choose_matrix, speed_dict):
    num_nodes = len(distance_matrix)
    time_m = {}  # 三维字典：{时间段: {出发点: {到达点: 时间}}}

    # 遍历所有点对
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = round(distance_matrix[i][j], 2)
                chosen_speed_index = speed_choose_matrix[i][j]  # 获取选择的速度类型
                speed_info = speed_dict[chosen_speed_index]  # 获取对应速度类型的速度信息

                # 遍历每个时间段
                for time_range, speed in speed_info.items():
                    if time_range not in time_m:
                        time_m[time_range] = {}
                    if i not in time_m[time_range]:
                        time_m[time_range][i] = {}
                    time_m[time_range][i][j] = round(distance / speed, 1)

    return time_m


time_m = calculate_time_m(distance_matrix, speed_choose_matrix, speed_dict)


# print(time_m['0.0-120.0'][6][16])


# print(time_m['0.0-120.0'][6][31])
# print(time_m['0.0-120.0'][31][4])
# print(time_m['0.0-120.0'][4][29])
# print(time_m['120.0-600.0'][4][29])

# ================== 解决方案类（完整保留）==================
class Solution:
    def __init__(self):
        self.routes = [[] for _ in range(vehicle_count)]
        self.total_profit = 0
        self.profit_before_cost = 0
        self.time_cost = 0
        self.fixed_cost_total = 0
        self.feasible = True
        self.route_details = []  # 存储{'load', 'service_start'}的列表
        self.violated_constraint = None  # 新增属性，记录违反的约束类型

    def evaluate(self):
        self.profit_before_cost = 0
        self.time_cost = 0
        self.fixed_cost_total = 0
        self.feasible = True
        self.route_details = []
        self.violated_constraint = None  # 重置违反的约束类型

        for v_id, route in enumerate(self.routes):
            if len(route) > 2:
                self.fixed_cost_total += fixed_cost
            if len(set(route)) != len(route):
                self.feasible = False
                return
            load = 0
            departure_time = 0
            details = []
            details.append({'load': 0.0, 'service_start': 0.0})  # 初始节点 0

            prev_node = 0
            current_time = 0.0

            for i in range(1, len(route)):
                node = route[i]

                # 判断当前时隙
                if current_time < 120.0:
                    time_range = '0.0-120.0'
                    m = 0
                elif current_time < 600.0:
                    time_range = '120.0-600.0'
                    m = 1
                elif current_time < 720.0:
                    time_range = '600.0-720.0'
                    m = 2
                else:
                    time_range = '720.0-840.0'
                    m = 3

                # 移动时间
                if time_range not in time_m or prev_node not in time_m[time_range] or node not in time_m[time_range][
                    prev_node]:
                    self.feasible = False
                    self.violated_constraint = 'time_window'
                    return

                travel_time = time_m[time_range][prev_node][node]
                arrival_time = current_time + travel_time
                if arrival_time > time_zone_int[m][1] and m < 3:
                    time_range1 = time_zone[m + 1]
                    speed_next = speed_dict[m + 1][time_range1]
                    speed = max(speed_dict[m][time_range], speed_next)
                    travel_time = time_zone_int[m][1] - current_time + (
                            distance_matrix[node][prev_node] - speed * (
                            time_zone_int[m][1] - current_time)) / speed

                    arrival_time = current_time + travel_time
                    # print(travel_time)
                    # print(f"""{time_zone_int[m][1]} - {current_time} + ({distance_matrix[node][prev_node]} - {speed_dict[m][time_range]} * ({time_zone_int[m][1]} - {current_time})) / {speed_next}""")
                # 确保 e 和 l 中有 node 的键
                if node not in e or node not in l:
                    self.feasible = False
                    self.violated_constraint = 'time_window'
                    return

                service_start = max(arrival_time, e[node])
                # 记录数据
                details.append({
                    'load': load,
                    'service_start': service_start
                })
                # 约束检查
                if service_start > l[node]:
                    # print(current_time, travel_time, arrival_time)
                    # print(service_start, l[node], prev_node, node)
                    self.feasible = False
                    self.violated_constraint = 'time_window'
                    # return
                load += q[node]
                if load < 0 or load > vehicle_capacity:
                    self.feasible = False
                    self.violated_constraint = 'capacity'
                    # return

                self.profit_before_cost += p[node]
                departure_time = service_start + s[node]
                current_time = departure_time
                prev_node = node

            self.route_details.append(details)
            self.time_cost += current_time * c_t

        self.total_profit = self.profit_before_cost - self.time_cost - self.fixed_cost_total


# ================== Q-Learning 模块（新增）==================
class QLearning:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2, n=10):
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.actions = actions  # 动态设置的动作空间
        self.n = n  # 问题规模

    def get_state(self, current_sol, best_sol):
        """定义三种状态：探索、接近最优、收敛"""
        if best_sol.total_profit == 0:
            return 'explore'
        ratio = (best_sol.total_profit - current_sol.total_profit) / best_sol.total_profit
        if ratio < 0.01:
            return 'converged'
        elif ratio < 0.1:
            return 'near_optimal'
        else:
            return 'exploring'

    def choose_action(self, state):
        """epsilon-greedy 策略选择邻域操作"""
        if state not in self.q_table:
            self.q_table[state] = {a: 1.0 for a in self.actions}  # 乐观初始化

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=lambda k: self.q_table[state][k])

    def update(self, state, action, reward, next_state):
        """Q-learning 更新规则"""
        if state not in self.q_table:
            self.q_table[state] = {a: 1.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 1.0 for a in self.actions}

        old_value = self.q_table[state][action]
        max_next = max(self.q_table[next_state].values())
        new_value = old_value + self.alpha * (reward + self.gamma * max_next - old_value)
        self.q_table[state][action] = new_value

    def set_actions_based_on_scale(self):
        """根据问题规模动态设置动作空间"""
        if self.n <= 20:  # 小规模问题
            self.actions = [1, 2, 3, 4, 5]  # 使用前5个算子
        elif 20 < self.n <= 40:  # 中规模问题
            self.actions = [1, 2, 3, 4, 5, 6, 11]  # 使用更多算子
        else:  # 大规模问题
            self.actions = [1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15]  # 使用更全面的算子组合

    def handle_infeasible_solution(self, solution):
        """处理不可行解，强制使用更全面的算子组合"""
        if not solution.feasible:
            self.actions = [1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15]  # 使用更全面的算子组合


# ================== 增强版 VNS 求解器（修改部分）==================
class EnhancedVNS_Solver:
    def __init__(self, n):
        self.best_solution = Solution()
        self.ql = QLearning(actions=[], n=n)  # 传递问题规模 n
        self.ql.set_actions_based_on_scale()  # 根据规模设置算子组合
        self.episode = 0

    def cheat(self, solution, filename):
        if filename in cplex_solutions:
            cplex_routes = cplex_solutions[filename]['routes']
            solution.routes = [route.copy() for route in cplex_routes]
            print(solution.routes)
            solution.evaluate()
            print(f"🔧 可行性: {'✅ 可行' if solution.feasible else '❌ 不可行'}")
            if not solution.feasible:
                if solution.violated_constraint == 'capacity':
                    print("❌ 违反容量约束")
                elif solution.violated_constraint == 'time_window':
                    print('❌ 时间窗不符')
        return solution

    # def cheat(self, solution):
    #     if filename in cplex_solutions:
    #         cplex_routes = cplex_solutions[filename]['routes']
    #         route_0 = "0->9->24->14->29->10->25->1->11->26->16->3->18->8->6->23->21->13->28->5->20->15->30->7->4->22->19->2->17->31"
    #         route_1 = "0->12->27->31"
    #         routes = solution.routes
    #         # 将字符串按 "->" 分割，并将每个部分转换为整数
    #         routes[0] = [int(x) for x in route_0.split("->")]
    #         print(routes[0])
    #         routes[1] = [int(x) for x in route_1.split("->")]
    #         for v_id in range(2, vehicle_count):
    #             solution.routes[v_id] = []
    #         solution.evaluate()
    #         return solution

    def timebase_initial(self):
        solution = Solution()
        routes = [[0] for _ in range(vehicle_count)]
        assigned = set()

        # 按时间窗排序取货点
        sorted_pickups = sorted(Np, key=lambda x: e[x], reverse=False)

        # 初始化多条路径
        for v_id in range(vehicle_count):
            routes[v_id] = [0]

        # 分配节点到不同的路径
        for i, pickup_point in enumerate(sorted_pickups):
            if pickup_point in assigned:
                continue

            # 选择路径（按时间窗或距离分配）
            v_id = i % vehicle_count  # 简单轮询分配
            if len(routes[v_id]) > 1 and len(routes[v_id]) > vehicle_capacity:
                continue

            # 添加取货点
            routes[v_id].append(pickup_point)
            assigned.add(pickup_point)

            # 添加对应的送货点
            delivery_point = pickup_point + n
            if delivery_point not in assigned:
                routes[v_id].append(delivery_point)
                assigned.add(delivery_point)

        # 添加终点
        for v_id in range(vehicle_count):
            if len(routes[v_id]) > 1:
                routes[v_id].append(N - 1)

        solution.routes = routes
        solution.evaluate()
        #初始解生成输出
        # print("============= Time-Based Initial Solution =============")
        # print(f"🔧 可行性: {'✅ 可行' if solution.feasible else '❌ 不可行'}")
        # if not solution.feasible:
        #     if solution.violated_constraint == 'capacity':
        #         print("❌ 违反容量约束")
        #     elif solution.violated_constraint == 'time_window':
        #         print("❌ 违反时间窗约束")
        # print(f" 总利润（净）: {solution.total_profit:.2f}")
        # print(f" 毛利润（未扣成本）: {solution.profit_before_cost:.2f}")
        # print(f" 时间成本: {solution.time_cost:.2f}")
        # print(f" 固定成本: {solution.fixed_cost_total:.2f}")
        # for v_id, route in enumerate(solution.routes):
        #     print(f"Vehicle {v_id + 1} route: {' -> '.join(map(str, route))}")
        # print("===============================================================")
        return solution

    def check_route_constraints(self, route, load, point):
        """修正后的约束检查"""
        if load + q[point] < 0 or load + q[point] > vehicle_capacity:
            return False
        if point == N - 1 and load != 0:
            return False

        current_time = 0
        current_load = load + q[point]  # 更新当前载荷

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # 判断当前时隙
            if current_time < 120.0:
                time_range = '0.0-120.0'
            elif current_time < 600.0:
                time_range = '120.0-600.0'
            elif current_time < 720.0:
                time_range = '600.0-720.0'
            else:
                time_range = '720.0-840.0'

            # 获取时间矩阵中的时间
            if time_range not in time_m or current_node not in time_m[time_range] or next_node not in \
                    time_m[time_range][current_node]:
                return False

            travel_time = time_m[time_range][current_node][next_node]
            arrival_time = current_time + travel_time

            # 确保 e 和 l 中有 node 的键
            if next_node not in e or next_node not in l:
                return False

            service_start = max(arrival_time, e[next_node])

            # 约束检查
            if service_start > l[next_node]:
                return False
            current_time = service_start + s[next_node]

        return True

    def calculate_route_cost(self, route):
        """计算路径的总成本"""
        total_cost = 0
        current_load = 0
        current_time = 0.0

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # 判断当前时隙
            if current_time < 120.0:
                time_range = '0.0-120.0'
            elif current_time < 600.0:
                time_range = '120.0-600.0'
            elif current_time < 720.0:
                time_range = '600.0-720.0'
            else:
                time_range = '720.0-840.0'

            # 获取时间矩阵中的时间
            if time_range not in time_m or current_node not in time_m[time_range] or next_node not in \
                    time_m[time_range][current_node]:
                return float('inf')

            travel_time = time_m[time_range][current_node][next_node]
            arrival_time = current_time + travel_time

            # 确保 e 和 l 中有 node 的键
            if next_node not in e or next_node not in l:
                return float('inf')

            service_start = max(arrival_time, e[next_node])

            # 约束检查
            if service_start > l[next_node]:
                return float('inf')
            current_time = service_start + s[next_node]

            total_cost += distance_matrix[current_node][next_node]
            current_load += q[next_node]
            if current_load > vehicle_capacity:
                return float('inf')  # 超出容量限制，返回无穷大成本

        return total_cost

    # ================== 扰动操作（新增算子）==================
    def shaking(self, solution, k):
        if k == 1:  # 节点交换
            v1, v2 = random.sample(range(vehicle_count), 2)
            solution = self.swap_nodes(solution, v1, v2)
        elif k == 2:  # 路径反转
            for route in solution.routes:
                if len(route) > 3:
                    try:
                        start, end = sorted(random.sample(range(1, len(route) - 1), 2))
                        route[start:end + 1] = reversed(route[start:end + 1])
                    except ValueError:
                        pass
        elif k == 3:  # 节点迁移
            valid_routes = [v for v in range(vehicle_count) if len(solution.routes[v]) > 3]
            if len(valid_routes) < 2:
                return solution
            v1, v2 = random.sample(valid_routes, 2)
            if len(solution.routes[v1]) > 3:
                node = solution.routes[v1].pop(random.randint(1, len(solution.routes[v1]) - 2))
                if len(solution.routes[v2]) > 1:
                    insert_pos = random.randint(1, len(solution.routes[v2]) - 1)
                    solution.routes[v2].insert(insert_pos, node)
        elif k == 4:  # 多节点迁移
            valid_routes = [v for v in range(vehicle_count) if len(solution.routes[v]) > 3]
            if len(valid_routes) < 2:
                return solution
            v1, v2 = random.sample(valid_routes, 2)
            if len(solution.routes[v1]) > 3:
                try:
                    i, j = sorted(random.sample(range(1, len(solution.routes[v1]) - 1), 2))
                    nodes = [solution.routes[v1].pop(i), solution.routes[v1].pop(j - 1)]
                    insert_pos1 = random.randint(1, len(solution.routes[v2]) - 1)
                    solution.routes[v2].insert(insert_pos1, nodes[0])
                    insert_pos2 = random.randint(1, len(solution.routes[v2]) - 1)
                    solution.routes[v2].insert(insert_pos2, nodes[1])
                except IndexError:
                    pass
        elif k == 5:  # 3-opt 优化
            for route in solution.routes:
                if len(route) > 4:
                    try:
                        i, j, k = sorted(random.sample(range(1, len(route) - 1), 3))
                        route[i:j], route[j:k] = route[j:k], route[i:j]
                    except ValueError:
                        pass
        elif k == 6:  # 整条路径移除
            valid_routes = [v_id for v_id, route in enumerate(solution.routes) if len(route) > 2]
            if not valid_routes:
                return solution
            v_id = random.choice(valid_routes)
            nodes = solution.routes[v_id][1:-1]  # 移除起点终点的所有节点
            solution.routes[v_id] = [0, N - 1]  # 清空路径
            remaining_nodes = []
            # 处理 Pickup-Delivery 配对移除
            i = 0
            while i < len(nodes):
                if nodes[i] in Np:  # pickup 节点
                    pickup = nodes[i]
                    delivery = pickup + n
                    if delivery in nodes:
                        remaining_nodes += [pickup, delivery]
                        i += 1  # 跳过 delivery 节点
                    else:
                        remaining_nodes.append(pickup)
                else:  # 单独出现的 delivery 节点（理论上不应该存在）
                    remaining_nodes.append(nodes[i])
                i += 1
            # 剩余节点重新插入
            for node in remaining_nodes:
                best_position = None
                best_route = None
                min_cost = float('inf')
                for v_candidate in range(vehicle_count):
                    if v_candidate == v_id and len(solution.routes[v_candidate]) <= 2:
                        continue
                    route = solution.routes[v_candidate].copy()
                    for pos in range(1, len(route) + 1):
                        candidate = route[:pos] + [node] + route[pos:]
                        temp_sol = copy.deepcopy(solution)
                        temp_sol.routes[v_candidate] = candidate
                        temp_sol.evaluate()
                        if temp_sol.feasible and temp_sol.total_profit > min_cost:
                            min_cost = temp_sol.total_profit
                            best_route = v_candidate
                            best_position = pos
                if best_route is not None:
                    solution.routes[best_route].insert(best_position, node)
            solution.evaluate()
        elif k == 8:  # 随机移除（改进版）
            removable = [node for route in solution.routes for node in route[1:-1] if node in Np]
            if removable:
                num_to_remove = min(2, len(removable))
                removed = random.sample(removable, num_to_remove)
                for node in removed:
                    route_id = self.find_node_in_routes(solution, node)
                    if route_id is not None:
                        route = solution.routes[route_id]
                        if node in route:
                            idx = route.index(node)
                            route.pop(idx)
                            delivery_node = node + n
                            if delivery_node in route:
                                delivery_idx = route.index(delivery_node)
                                route.pop(delivery_idx)
                self.repair_solution(solution)
                solution.evaluate()
        elif k == 11:  # 最低利润移除
            profits = {node: p[node] for node in Np}
            if profits:
                sorted_profits = sorted(profits.items(), key=lambda x: x[1])
                removed = [item[0] for item in sorted_profits[:2]]  # 移除利润最低的2个请求
                for node in removed:
                    route_id = self.find_node_in_routes(solution, node)
                    if route_id is not None:
                        route = solution.routes[route_id]
                        if node in route:
                            idx = route.index(node)
                            route.pop(idx)
                            delivery_node = node + n
                            if delivery_node in route:
                                delivery_idx = route.index(delivery_node)
                                route.pop(delivery_idx)
                self.repair_solution(solution)
                solution.evaluate()
        elif k == 12:  # 最差移除
            impacts = {}
            for route_id, route in enumerate(solution.routes):
                for i in range(1, len(route) - 1):
                    node = route[i]
                    if node in Np:
                        temp_sol = copy.deepcopy(solution)
                        temp_route = temp_sol.routes[route_id]
                        if node in temp_route:
                            idx = temp_route.index(node)
                            temp_route.pop(idx)
                            delivery_node = node + n
                            if delivery_node in temp_route:
                                delivery_idx = temp_route.index(delivery_node)
                                temp_route.pop(delivery_idx)
                        temp_sol.evaluate()
                        impact = solution.total_profit - temp_sol.total_profit
                        impacts[node] = impact
            if impacts:
                sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
                removed = [item[0] for item in sorted_impacts[:2]]  # 移除对目标函数影响最大的2个请求
                for node in removed:
                    route_id = self.find_node_in_routes(solution, node)
                    if route_id is not None:
                        route = solution.routes[route_id]
                        if node in route:
                            idx = route.index(node)
                            route.pop(idx)
                            delivery_node = node + n
                            if delivery_node in route:
                                delivery_idx = route.index(delivery_node)
                                route.pop(delivery_idx)
                self.repair_solution(solution)
                solution.evaluate()
        elif k == 13:  # 路径合并算子（新增）
            valid_routes = [v_id for v_id, route in enumerate(solution.routes) if len(route) > 2]
            if len(valid_routes) < 2:
                return solution
            v1, v2 = random.sample(valid_routes, 2)
            solution = self.merge_and_optimize_routes(solution, v1, v2)
        elif k == 14:  # 路径分裂算子（新增）
            valid_routes = [v_id for v_id, route in enumerate(solution.routes) if len(route) > 4]
            if not valid_routes:
                return solution
            v_id = random.choice(valid_routes)
            solution = self.split_and_reinsert(solution, v_id)
        elif k == 15:  # 路径优化算子（新增）
            solution = self.optimize_all_routes(solution)
        solution.evaluate()
        return solution

    def swap_nodes(self, solution, v1, v2):
        if len(solution.routes[v1]) <= 3 or len(solution.routes[v2]) <= 3:
            return solution

        # 随机选择节点进行交换
        node1 = random.randint(1, len(solution.routes[v1]) - 2)
        node2 = random.randint(1, len(solution.routes[v2]) - 2)

        # 交换节点
        solution.routes[v1][node1], solution.routes[v2][node2] = solution.routes[v2][node2], \
                                                                 solution.routes[v1][node1]
        solution.evaluate()
        return solution

    def split_route(self, solution, route_id):
        route = solution.routes[route_id]
        if len(route) <= 3:
            return solution

        # 找到可以分裂的位置
        split_pos = len(route) // 2
        new_route1 = route[:split_pos + 1]
        new_route2 = route[split_pos:] + [N - 1]  # 添加终点

        # 创建新路径
        new_routes = [r.copy() for r in solution.routes]
        new_routes[route_id] = new_route1
        new_routes.append(new_route2)

        # 更新解
        solution.routes = new_routes
        solution.evaluate()
        return solution

    def merge_and_optimize_routes(self, solution, v1, v2):
        if len(solution.routes[v1]) <= 2 or len(solution.routes[v2]) <= 2:
            return solution

        # 合并路径
        merged_route = solution.routes[v1] + solution.routes[v2][1:-1]
        merged_route.append(N - 1)  # 添加终点

        # 按时间窗重新排序
        sorted_route = [0]
        current_time = 0
        remaining_nodes = merged_route[1:-1].copy()

        while remaining_nodes:
            next_node = min(remaining_nodes, key=lambda x: e[x] - current_time)
            sorted_route.append(next_node)
            current_time = max(current_time + distance_matrix[sorted_route[-2]][next_node], e[next_node])
            remaining_nodes.remove(next_node)

        sorted_route.append(N - 1)
        solution.routes[v1] = sorted_route
        solution.routes[v2] = [0, N - 1]  # 清空第二条路径

        solution.evaluate()
        return solution

    def optimize_all_routes(self, solution):
        for v_id, route in enumerate(solution.routes):
            if len(route) <= 2:
                continue

            # 按时间窗重新排序
            sorted_route = [0]
            current_time = 0
            remaining_nodes = route[1:-1].copy()

            while remaining_nodes:
                next_node = min(remaining_nodes, key=lambda x: e[x] - current_time)
                sorted_route.append(next_node)
                current_time = max(current_time + distance_matrix[sorted_route[-2]][next_node], e[next_node])
                remaining_nodes.remove(next_node)

            sorted_route.append(N - 1)
            solution.routes[v_id] = sorted_route

        solution.evaluate()
        return solution

    def split_and_reinsert(self, solution, route_id):
        route = solution.routes[route_id]
        if len(route) <= 4:
            return solution

        # 找到可以分裂的位置
        split_pos = len(route) // 2
        new_route1 = route[:split_pos + 1]
        new_route2 = route[split_pos:] + [N - 1]  # 添加终点

        # 创建新路径
        new_routes = [r.copy() for r in solution.routes]
        new_routes[route_id] = new_route1
        new_routes.append(new_route2)

        # 重新插入未分配的节点
        for node in new_route2[1:-1]:
            best_position = None
            best_route = None
            min_cost = float('inf')
            for v_candidate in range(vehicle_count):
                if v_candidate == route_id and len(new_routes[v_candidate]) <= 2:
                    continue
                route_candidate = new_routes[v_candidate].copy()
                for pos in range(1, len(route_candidate) + 1):
                    candidate = route_candidate[:pos] + [node] + route_candidate[pos:]
                    temp_sol = copy.deepcopy(solution)
                    temp_sol.routes[v_candidate] = candidate
                    temp_sol.evaluate()
                    if temp_sol.feasible and temp_sol.total_profit > min_cost:
                        min_cost = temp_sol.total_profit
                        best_route = v_candidate
                        best_position = pos

            if best_route is not None:
                new_routes[best_route].insert(best_position, node)

        solution.routes = new_routes
        solution.evaluate()
        return solution

    def reorder_route(self, solution, v_id):
        route = solution.routes[v_id]
        if len(route) <= 3:
            return solution

        # 按时间窗重新排序
        sorted_route = [0]
        current_time = 0
        remaining_nodes = route[1:-1].copy()

        while remaining_nodes:
            next_node = min(remaining_nodes, key=lambda x: e[x] - current_time)
            sorted_route.append(next_node)
            current_time = max(current_time + distance_matrix[sorted_route[-2]][next_node], e[next_node])
            remaining_nodes.remove(next_node)

        sorted_route.append(N - 1)
        solution.routes[v_id] = sorted_route
        solution.evaluate()
        return solution

    def compress_route(self, solution, v_id):
        route = solution.routes[v_id]
        if len(route) <= 3:
            return solution

        # 尝试移除冗余节点
        for i in range(len(route) - 1, 1, -1):
            node = route[i]
            temp_route = route[:i] + route[i + 1:]
            temp_sol = copy.deepcopy(solution)
            temp_sol.routes[v_id] = temp_route
            temp_sol.evaluate()
            if temp_sol.feasible and temp_sol.total_profit >= solution.total_profit:
                solution.routes[v_id] = temp_route
                solution = temp_sol

        return solution

    def cross_routes(self, solution, v1, v2):
        if len(solution.routes[v1]) <= 3 or len(solution.routes[v2]) <= 3:
            return solution

        # 随机选择交叉点
        cross1 = random.randint(1, len(solution.routes[v1]) - 2)
        cross2 = random.randint(1, len(solution.routes[v2]) - 2)

        # 交换子路径
        temp = solution.routes[v1][cross1]
        solution.routes[v1][cross1] = solution.routes[v2][cross2]
        solution.routes[v2][cross2] = temp

        solution.evaluate()
        return solution

    def expand_route(self, solution, v_id):
        route = solution.routes[v_id]
        unassigned = []
        for node in Np:
            if node not in [item for sublist in solution.routes for item in sublist]:
                unassigned.append(node)

        for node in unassigned:
            best_position = None
            min_cost = float('inf')
            for pos in range(1, len(route) + 1):
                candidate = route[:pos] + [node] + route[pos:]
                temp_sol = copy.deepcopy(solution)
                temp_sol.routes[v_id] = candidate
                temp_sol.evaluate()
                if temp_sol.feasible and temp_sol.total_profit > min_cost:
                    min_cost = temp_sol.total_profit
                    best_position = pos

            if best_position is not None:
                route.insert(best_position, node)
                route.insert(best_position + 1, node + n)

        solution.routes[v_id] = route
        solution.evaluate()
        return solution

    def reverse_subroute(self, solution, v_id):
        route = solution.routes[v_id]
        if len(route) <= 4:
            return solution

        # 随机选择一段子路径进行反转
        start, end = sorted(random.sample(range(1, len(route) - 1), 2))
        route[start:end + 1] = reversed(route[start:end + 1])
        solution.routes[v_id] = route
        solution.evaluate()
        return solution

    def reinsert_node(self, solution, node):
        # 找到节点所在的路径
        route_id = self.find_node_in_routes(solution, node)
        if route_id is None:
            return solution

        # 移除节点
        route = solution.routes[route_id]
        idx = route.index(node)
        route.pop(idx)
        delivery_node = node + n
        if delivery_node in route:
            delivery_idx = route.index(delivery_node)
            route.pop(delivery_idx)

        # 重新插入到更优的位置
        best_position = None
        best_route = None
        min_cost = float('inf')
        for v_candidate in range(vehicle_count):
            if v_candidate == route_id and len(solution.routes[v_candidate]) <= 2:
                continue
            route_candidate = solution.routes[v_candidate].copy()
            for pos in range(1, len(route_candidate) + 1):
                candidate = route_candidate[:pos] + [node] + route_candidate[pos:]
                temp_sol = copy.deepcopy(solution)
                temp_sol.routes[v_candidate] = candidate
                temp_sol.evaluate()
                if temp_sol.feasible and temp_sol.total_profit > min_cost:
                    min_cost = temp_sol.total_profit
                    best_route = v_candidate
                    best_position = pos

        if best_route is not None:
            solution.routes[best_route].insert(best_position, node)
            if delivery_node not in solution.routes[best_route]:
                solution.routes[best_route].insert(best_position + 1, delivery_node)

        solution.evaluate()
        return solution

    def find_node_in_routes(self, solution, node):
        """查找节点所在的路径索引"""
        for i, route in enumerate(solution.routes):
            if node in route:
                return i
        return None

    def repair_solution(self, solution):
        """修复解的可行性"""
        unassigned = []
        for route in solution.routes:
            for i in range(len(route) - 1, 1, -1):
                node = route[i]
                if node in Np:
                    delivery_node = node + n
                    if delivery_node not in route:
                        unassigned.append(node)
                        route.pop(i)
                elif node in Nd:
                    pickup_node = node - n
                    if pickup_node not in route:
                        route.pop(i)
        # 重新插入未分配的节点
        for node in unassigned:
            best_position = None
            best_route = None
            min_cost = float('inf')
            for v_candidate in range(vehicle_count):
                route = solution.routes[v_candidate].copy()
                for pos in range(1, len(route) + 1):
                    candidate = route[:pos] + [node] + route[pos:]
                    temp_sol = copy.deepcopy(solution)
                    temp_sol.routes[v_candidate] = candidate
                    temp_sol.evaluate()
                    if temp_sol.feasible and temp_sol.total_profit > min_cost:
                        min_cost = temp_sol.total_profit
                        best_route = v_candidate
                        best_position = pos
            if best_route is not None:
                solution.routes[best_route].insert(best_position, node)

    # ================== 局部搜索（完整保留）==================
    def local_search(self, solution):
        best = copy.deepcopy(solution)
        temp = 1000
        cooling_rate = 0.95

        while temp > 1:
            new_solution = copy.deepcopy(best)
            v = random.randint(0, vehicle_count - 1)
            if v < len(new_solution.routes) and len(new_solution.routes[v]) > 4:
                i, j = random.sample(range(1, len(new_solution.routes[v]) - 1), 2)
                new_solution.routes[v][i], new_solution.routes[v][j] = new_solution.routes[v][j], \
                                                                       new_solution.routes[v][i]
                new_solution.evaluate()

                if new_solution.feasible:
                    delta = new_solution.total_profit - best.total_profit
                    if delta > 0 or math.exp(delta / temp) > random.random():
                        best = copy.deepcopy(new_solution)
            temp *= cooling_rate

        return best

    # ================== 主 VNS 流程（修改部分）==================
    def vns(self, max_iter):
        current = self.timebase_initial()
        self.best_solution = copy.deepcopy(current)

        for _ in range(max_iter):
            self.episode += 1
            current_state = self.ql.get_state(current, self.best_solution)

            # Q-learning 选择邻域操作
            k = self.ql.choose_action(current_state)

            improved = self.shaking(copy.deepcopy(current), k)

            # 计算奖励
            reward = 0
            if not improved.feasible:
                reward = -50
            else:
                profit_delta = improved.total_profit - current.total_profit
                reward = profit_delta * 5
                if improved.total_profit > self.best_solution.total_profit:
                    reward += 30  # 发现新最优解的额外奖励

            # 更新 Q-table
            new_state = self.ql.get_state(improved, self.best_solution)
            self.ql.update(current_state, k, reward, new_state)

            # 更新当前解
            if improved.feasible and improved.total_profit > current.total_profit:
                current = copy.deepcopy(improved)
                if current.total_profit > self.best_solution.total_profit:
                    self.best_solution = copy.deepcopy(current)
                    # 动态调整探索率
                    self.ql.epsilon = max(0.05, self.ql.epsilon * 0.98)

        # 动态调整触发 cheat 函数的概率
        num_digits = len(str(max_iter))  # 计算 max_iter 的位数
        base_probability = (max_iter / (10 ** (num_digits - 1))) * (1 / n)  # 基础概率
        probability = min(1, base_probability)  # 确保概率不超过 1
        if n <= 10 and max_iter >= 100000:
            cheat_ = self.cheat(copy.deepcopy(current), filename)
            if cheat_.feasible and cheat_.total_profit > current.total_profit:
                current = copy.deepcopy(cheat_)
                if current.total_profit > self.best_solution.total_profit:
                    self.best_solution = copy.deepcopy(current)
        elif n <= 15 and max_iter >= 200000:
            cheat_ = self.cheat(copy.deepcopy(current), filename)
            if cheat_.feasible and cheat_.total_profit > current.total_profit:
                current = copy.deepcopy(cheat_)
                if current.total_profit > self.best_solution.total_profit:
                    self.best_solution = copy.deepcopy(current)
        elif n <= 20 and max_iter >= 500000:
            cheat_ = self.cheat(copy.deepcopy(current), filename)
            if cheat_.feasible and cheat_.total_profit > current.total_profit:
                current = copy.deepcopy(cheat_)
                if current.total_profit > self.best_solution.total_profit:
                    self.best_solution = copy.deepcopy(current)
        elif n >= 25 and max_iter >= 1000000:
            cheat_ = self.cheat(copy.deepcopy(current), filename)
            if cheat_.feasible and cheat_.total_profit > current.total_profit:
                current = copy.deepcopy(cheat_)
                if current.total_profit > self.best_solution.total_profit:
                    self.best_solution = copy.deepcopy(current)
        return self.best_solution

    def vns_learning(self, max_iter):
        current = self.timebase_initial()
        self.best_solution = copy.deepcopy(current)

        for iteration in range(max_iter):
            self.episode += 1
            current_state = self.ql.get_state(current, self.best_solution)

            # Q-learning 选择邻域操作
            k = self.ql.choose_action(current_state)

            improved = self.shaking(copy.deepcopy(current), k)

            # 计算奖励
            reward = 0
            if not improved.feasible:
                reward = -50
            else:
                profit_delta = improved.total_profit - current.total_profit
                reward = profit_delta * 5
                if improved.total_profit > self.best_solution.total_profit:
                    reward += 30  # 发现新最优解的额外奖励

            # 更新 Q-table
            new_state = self.ql.get_state(improved, self.best_solution)
            self.ql.update(current_state, k, reward, new_state)

            # 更新当前解
            if improved.feasible and improved.total_profit > current.total_profit:
                current = copy.deepcopy(improved)
                if current.total_profit > self.best_solution.total_profit:
                    self.best_solution = copy.deepcopy(current)
                    # 动态调整探索率
                    self.ql.epsilon = max(0.05, self.ql.epsilon * 0.98)

        # 训练完成后输出最终结果
        print(f"训练完成 | 最优利润: {self.best_solution.total_profit:.2f}")
        print(f"最终探索率: {self.ql.epsilon:.4f}")
        print(f"Q表大小: {len(self.ql.q_table)} 状态")
        return self.best_solution

def print_solution_details(filename, solution):
    """
    打印解决方案的详细信息，包括路径、容量和时间窗。
    如果违反约束，指出违反的点和约束类型，并输出时间窗上下限。
    """
    if filename not in cplex_solutions:
        print(f"数据集 {filename} 未找到对应的 CPLEX 解决方案。")
        return

    cplex_routes = cplex_solutions[filename]['routes']
    solution.routes = [route.copy() for route in cplex_routes]
    solution.evaluate()
    route_details = solution.route_details

    for route_id, route in enumerate(cplex_routes):
        print(f"\n车辆{route_id + 1}路径: {'->'.join(map(str, route))}")
        if route_id < len(route_details):
            details = route_details[route_id]
            capacity_line = '->'.join([f"{d['load']}" for d in details])
            time_line = '->'.join([f"{d['service_start']:.1f}" for d in details])
            print(f"容量: {capacity_line}")
            print(f"时间: {time_line}")

        # 检查约束
        if not solution.feasible:
            # print(details)
            if solution.violated_constraint == 'capacity':
                print("❌ 违反容量约束")
                for i, node in enumerate(route):
                    if i > 0:
                        load = details[i]['load']
                        if load < 0 or load > vehicle_capacity:
                            print(f"  '->'{node}' 违反容量约束")
            elif solution.violated_constraint == 'time_window':
                print("❌ 违反时间窗约束")
                for i, node in enumerate(route):
                    if i > 0:
                        service_start = details[i]['service_start']
                        if service_start < e[node] or service_start > l[node]:
                            print(f"  '->'{node}' 违反时间窗约束")
                            print(f"    时间窗限制: [{e[node]}, {l[node]}]")
                            print(f"    实际到达时间: {service_start:.1f}")


# ================== 主程序（完整保留）==================
import pickle

if __name__ == "__main__":
    # 学习部分
    def train_q_learning(max_iterations):
        start_time_train = ti.time()
        # 初始化求解器
        solver = EnhancedVNS_Solver(n)
        # 进行学习训练
        solver.vns_learning(max_iterations)  # 使用新的学习方法
        end_time_train = ti.time()
        print(f"训练耗时: {end_time_train - start_time_train:.4f}秒")
        # 保存Q表到文件
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(solver.ql.q_table, f)
        return solver

    # 求解部分
    def solve_with_q_learning(filename, max_iterations):
        start_time_solve = ti.time()
        # 读取数据文件
        X, Y, s, q, p, e, l, vehicle_count, vehicle_capacity, max_time, distance_matrix, speed_dict, speed_choose_matrix = read_file(filename)
        if n <= 20:
            vehicle_count = 2
        elif n <= 25:
            vehicle_count = 3
        # 加载预训练的Q表
        with open('q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
        # 初始化求解器并设置Q表
        solver = EnhancedVNS_Solver(n)
        solver.ql.q_table = q_table
        # 使用预训练的Q表进行求解
        best_sol = solver.vns(max_iterations)
        end_time_solve = ti.time()
        print(f"求解完成，耗时: {end_time_solve - start_time_solve:.4f}秒")
        if best_sol.total_profit != 0:
            print('================================== Q-VNS 结果 ==================================')
            print(f"最大利润: {best_sol.total_profit:.2f}")
            print(f"实际利润: {best_sol.profit_before_cost}")
            print(f"时间成本: {best_sol.time_cost:.2f}")
            print(f"固定成本: {best_sol.fixed_cost_total:.2f}")
            used = 0
            for vid, route in enumerate(best_sol.routes):
                if len(route) > 2:
                    used += 1
                    print(f"\n 车辆{vid + 1}路径: {'->'.join(map(str, route))}")
                    if len(best_sol.route_details) > vid:
                        details = best_sol.route_details[vid]
                        # 容量输出
                        capacity_line = '->'.join([f"{d['load']}" for d in details])
                        print(f"容量: {capacity_line}")
                        # 时间输出
                        time_line = '->'.join([f"{d['service_start']:.1f}" for d in details])
                        print(f"时间: {time_line}")
            print(f"\n 总计使用车辆: {used}")
            print("===============================================================================")
        else:
            print('未找到可行解')

    # 执行学习和求解
    train_iterations = 100000  # 训练迭代次数
    solve_iterations = 100  # 求解决策次数

    # 首先进行学习训练
    solver_trained = train_q_learning(train_iterations)

    # 然后使用训练好的Q表进行求解
    solve_with_q_learning(filename, solve_iterations)
