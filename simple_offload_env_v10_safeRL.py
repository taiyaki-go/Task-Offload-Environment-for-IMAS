import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import csv

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

import numpy as np

######################################### システム全体に関わるパラメータの定義 ######################################
VEHICLE_NUM = 50
TERMINATE_TIME = 1000
N_LEARN = 10000 * VEHICLE_NUM
SAFETY_EVALUATION_TIMESTEP = 10

################################################ RND Wrapper ##################################################
# RNDを適用 -> 探索範囲の拡大
class RNDModel(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(RNDModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class RNDWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RNDWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target = RNDModel(obs_dim).to(self.device)
        self.predictor = RNDModel(obs_dim).to(self.device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-4)

        # freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

    def compute_intrinsic_reward(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            target_feat = self.target(obs_tensor)
        pred_feat = self.predictor(obs_tensor)
        intrinsic_reward = torch.norm(pred_feat - target_feat, dim=1)
        return intrinsic_reward.detach().cpu().numpy()

    def step(self, action):
        obs, ext_reward, done, truncated, info = self.env.step(action)
        intrinsic_reward = self.compute_intrinsic_reward(obs[None, :])[0]

        # update predictor network
        self.optimizer.zero_grad()
        loss = torch.norm(self.predictor(torch.tensor(obs[None, :], dtype=torch.float32).to(self.device)) -
                          self.target(torch.tensor(obs[None, :], dtype=torch.float32).to(self.device)), dim=1).mean()
        loss.backward()
        self.optimizer.step()

        total_reward = ext_reward + intrinsic_reward  # 合成報酬
        return obs, total_reward, done, truncated, info


######################################################## 正規化 Wrapper ####################################################
# 正規化のためのWrapper
# 注意：VecEnvで環境を作る場合はVecNormalize()があるため, このNormalizeObservationは不要.
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):

        observation_space = env.observation_space
        assert isinstance(
            env.observation_space, gym.spaces.Box
            ), "ObservationWrapper only works with Box spaces"
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high

        self.observation_space = gym.spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = observation_space.shape,
            dtype = np.float32
        )
        super(NormalizeObservation, self).__init__(env)

    def observation(self, obs):
        return (obs - self.obs_low) / (self.obs_high - self.obs_low + 1e-8)

class StandardizeReward(gym.RewardWrapper):
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 1e-4
        self.epsilon = epsilon

    def reward(self, reward):
        self.count += 1
        alpha = 1.0 / self.count
        self.running_mean += alpha * (reward - self.running_mean)
        self.running_var += alpha * ((reward - self.running_mean) ** 2 - self.running_var)
        std = (self.running_var + self.epsilon) ** 0.5

        return (reward - self.running_mean) / std


############################################## safe RL 不確実性予測モデル ###########################################
class SafetyUncertaintyModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(SafetyUncertaintyModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # 状態とアクションを結合したものを入力とする
        input_dim = obs_dim + action_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 平均 (mu) と分散 (log_var) を出力
        # 分散は非負である必要があるため、log_varとして出力し、後で exp(log_var) する
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        # one-hotエンコーディングされたアクションを観測に結合
        action_one_hot = torch.zeros(action.shape[0], self.action_dim, device=obs.device)
        action_one_hot.scatter_(1, action.long().unsqueeze(1), 1)
        
        x = torch.cat([obs, action_one_hot], dim=1)
        
        features = self.feature_extractor(x)
        mu = self.mu_head(features)
        log_var = self.log_var_head(features)
        
        # 分散は exp(log_var) で計算
        var = torch.exp(log_var)
        
        return mu, var # 平均と分散

########################################################## タスクの定義 (重要度は連続値) #########################################################
class Task():

    CRITICAL_TASK_RATE = 0.05
    ID = 0 # タスクのID. safeRLで緊急アクションをとったときのreplay buffer書き換えに使用.

    def __init__(self, v_num, dummy=False): # 時間とタスクを保持している車両ID, 高重要度タスクの生成確率を初期化の引数としてとる
        self.location = 0 # 0: タスクが到着した状態 / 1: local_tq 内に存在している状態 / 2: mec_tq内に存在している状態
        self.vehicle = v_num # この計算タスクを保持している車両Noを表す. 車両Noは1オリジンであることに注意.
        self.compute_size = random.uniform(0.3, 1.2) # タスクの計算量 (G cycles)
        self.max_delay = random.uniform(0.5, 1.0) # タスクの遅延制約 (sec)
        self.importance = random.uniform(0.001, 0.999) # タスクの重要度
        self.input_size = self.compute_size # タスクの入力データ量 (Mbit)
        self.utility = 0 # 処理の成功によるUtilityを記録する変数
        self.id = Task.ID # タスクのIDを設定
        Task.ID = Task.ID + 1 # IDを更新
        coin = random.random()

        if coin < self.CRITICAL_TASK_RATE:
            self.is_safe_critical = 1 # 安全に関わるタスクであることを示す.
        else:
            self.is_safe_critical = 0 # 安全に関わるタスクではないことを示す.

        if dummy == True: # 計算タスクを保持していないことを示す, ダミー計算タスクを要求された場合はダミー計算タスクを生成 (本質的には必要ないが, 安全なコードにするために実装)
            self.location = -1 # ダミー計算タスクのlocationは-1とする (存在しないことを示す)
            self.compute_size = 0 # ダミー計算タスクの計算量は0
            self.max_delay = 1000 # ダミー計算タスクの遅延制約には十分に長い時間を設定しておく
            self.is_safe_critical = 0 # 安全に関わらないタスクとして設定
        
############################################################## 車両の定義 ######################################################################
class Vehicle():
    def __init__(self, id): # 現在の時刻, 車両ID を初期化の引数としてとる
        self.id = id # 車両ID
        self.trajectory = [] # 車両の交通流データを格納するリスト. i番目の要素が時刻iでの座標[x,y]となる. 注意：str型として座標は記録されている.
        self.x = 1 # 車両の位置 (x座標) (初期値として1を与えているが, 実際はタイムスロットごとにセットする.)
        self.y = 1 # 車両の位置 (y座標) (初期値として1を与えているが, 実際はタイムスロットごとにセットする.)
        self.v_x = 0 # 車両の速度ベクトル (x成分) (現在は未使用. 将来的な利用を考慮して実装.)
        self.v_y = 0 # 車両の速度ベクトル (y成分) (現在は未使用. 将来的な利用を考慮して実装.)
        #self.createTask(p)
        self.local_tq = deque() # 車両でのローカル処理を選択した計算タスクがappendされていく車両タスクキュー
    
    def createTask(self, p): # 車両に計算タスクを発生させる. 発生確率pを引数にとる.
        coin = random.random() # タスクが発生するかどうかを決定するために使用
        if coin <= p: # 計算タスクが発生
            self.task = Task(self.id)
            self.exist_task = True
        else: # 計算タスクは発生しない
            self.task = Task(self.id, dummy=True) # タスク自体は発生させ, ダミー計算タスクを紐付けるが, exist_taskをFalseにしてタスクが存在しないことを示す
            self.exist_task = False
    
    def setLocation(self, t): # 引数として取った時刻tにおける車両の位置をセット
        # 注意：self.trajectory[t-1][0]には, SUMOのシミュレーション時間が格納されている.
        self.x = float(self.trajectory[t-1][1])
        self.y = float(self.trajectory[t-1][2])

############################################################## MECの定義 ######################################################################
class Mec():
    def __init__(self, id, x, y, r): # MEC ID, MECの位置(x,y), 通信可能範囲r を初期化の引数としてとる
        self.id = id # MEC ID
        self.x = x # MECの位置 (x座標)
        self.y = y # MECの位置 (y座標)
        self.range = r # MECの通信可能範囲
        self.mec_tq = deque() # オフロードされた計算タスクがappendされていくMECタスクキュー
        self.link = 0 # 通信リンクを張っている車両数

########################################################## タスクオフロード環境の定義 ###########################################################
class SimpleOffloadEnv(gym.Env):

    # オフロード戦略を表す変数を定義
    LOCAL = 0 # タスクが車両のリソースで処理されるオフロード戦略
    MEC = 1 # タスクがオフロードされ, MECのリソースで処理されるオフロード戦略

    ######################################################## 各種パラメータの定義 ##################################################

    ALPHA = 1.0 # 効用関数において, タスクの処理遅延に関する項とエネルギー消費に関する項のバランスを決定するパラメタ.
                # U = ALPHA*(処理遅延に関する項) + (1-ALPHA)*(エネルギー消費に関する項)
    BETA = VEHICLE_NUM * 10 # 長期的報酬と短期的報酬のバランスを決定するパラメタ. 長期的報酬はBETA倍される.
    REWARD = 1 # タスクを遅延制約内で処理成功したときの, 固定効用r
    PENALTY_SCALAR_C = 1.0
    SAFETY_THRESHOLD = 0.2

    # 注意 : 計算タスクの遅延制約や計算量などはclass:taskで定義されている.
    ARRIVAL_RATE = 0.6 # 車両に対する, 1タイムスロットでの計算タスクの発生確率

    NUM_CORE = 10 # MECのコア数
    F_MEC = 3.2 * NUM_CORE # (G cycles / s) # MECの計算能力
    F_VEHICLE = 0.8 # (G cycles / s) # 車両の計算能力
    VEHICLE_NUM = VEHICLE_NUM # システム内の車両数
    MEC_NUM = 1 # システム内のMEC数
    TERMINATE_TIME = TERMINATE_TIME # シミュレーション終了時間. 1から開始.

    # 通信に関するパラメタの設定
    WAVEGHz = 5.9 # 車両とRSUの無線通信の周波数. IEEE802.11pを仮定.
    BANDWIDTH = 10 # 車両とRSUの無線通信の帯域幅. (MHz)
    A_GAIN_v = 0 # 車両側のアンテナの絶対利得 (dBi)
    A_GAIN_r = 0 # RSU側のアンテナの絶対利得 (dBi)
    T_POWER = 0.1 # 送信電力 (W)
    NOISE = -97 # ノイズの強さ (dBm)

    ##############################################################################################################################


    def __init__(self, model=None, is_training=False):

        super(SimpleOffloadEnv, self).__init__()

        self.LOCAL_CAPACITY = 200 # local_tqの最大長を定義. (本質的には必要ないが, 観測空間を有限にするために定義. violateが起こらないような十分に長いキュー長にする必要あり.)
        self.MEC_CAPACITY = 200 # mec_tqの最大長を定義. (本質的には必要ないが, 観測空間を有限にするために定義. violateが起こらないような十分に長いキュー長にする必要あり.)

        self.TASK_C_MAX = (2.0 * 1000) / 1000 # 計算タスクの計算量の最大値を定義
        self.TASK_DELAY_MAX = 1.0
        self.TASK_IMPORTANCE_MAX = 1 # 計算タスクが持つ重要度の最大値を定義

        self.LOCAL_TQ_CALMAX = self.LOCAL_CAPACITY * self.TASK_C_MAX / self.F_VEHICLE # local_tq に並んでいるタスクの計算にかかる時間の最大値
        self.MEC_TQ_CALMAX = self.MEC_CAPACITY * self.TASK_C_MAX / self.F_MEC # mec_tq に並んでいるタスクの計算にかかる時間の最大値

        # 強化学習モデルを環境に紐付けておく
        self.model = model

        # 学習のための環境構築かどうかを紐付けておく
        self.is_training = is_training

        # 行動空間 (action space) を定義
        n_actions = 2 # アクション数 (localでの処理, MECへのオフロード)
        self.action_space = gym.spaces.Discrete(n_actions) # アクションは離散的なのでDiscrete. 連続的であればBoxで定義.

        # 観測空間 (observation space) を定義
        # 観測は, (localでの処理にかかる処理時間, mecへのオフロードでかかる処理時間, 計算タスクの計算量, 計算タスクの遅延制約, 計算タスクの重要度, 安全に関わるタスクかどうか)
        low_lim = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32) # 観測空間の最低値
        high_lim = np.array([self.LOCAL_TQ_CALMAX, self.MEC_TQ_CALMAX, self.TASK_C_MAX, self.TASK_DELAY_MAX, self.TASK_IMPORTANCE_MAX, 1], dtype=np.float32) # 観測空間の最大値
        self.observation_space = gym.spaces.Box(
            low=low_lim, high=high_lim, shape=(6,), dtype=np.float32
        )

        # Safe RLのためのモデルと最適化器の初期化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.safety_uncertainty_model = SafetyUncertaintyModel(
            obs_dim=self.observation_space.shape[0], 
            action_dim=self.action_space.n
        ).to(self.device)
        self.safety_optimizer = optim.Adam(self.safety_uncertainty_model.parameters(), lr=1e-4)
        
        # 安全コストの学習データを格納するバッファ
        self.safety_replay_buffer = deque(maxlen=10000) 
        
        # 論文の b_h に相当する安全閾値。重要度の高いタスクが失敗した際の最大許容安全コスト。
        # 例えば、重要度0.8以上のタスクは、安全コスト0.1を超えてはならない、などと設定。
        # ここでは、重要度1のタスクが遅延制約を超えた場合に1の安全コストが発生すると仮定し、
        # 許容できる安全コストの閾値を非常に小さく設定することで、失敗を強く回避させる。
        self.safety_threshold_b_h = self.SAFETY_THRESHOLD # 安全コストがこれを超えないことを目標とする

        # システム内変数の初期化
        self.time = 1 # システム内の時刻
        self.vehicle_q = deque() # vehicle_q は, システム内の車両のリスト
        for i in range(self.VEHICLE_NUM):
            self.vehicle_q.append(Vehicle(i+1)) # 車両をVEHICLE_NUMだけ生成し, vehicle_qにappendしていく.
        task_create = False
        while task_create == False:
            task_create = self.createTaskForVehicle()
        self.vehicle_trajectory = [] # 車両の交通流データを格納するリスト. def loadVehicleTrajectory でこのリストに交通流データを格納していく.
        self.processed_vehicle_num = 0 # 現在のタイムスロットでオフロード戦略の決定が完了した車両数(processed_vehicle_num)を0として初期化
        self.mec_q = deque() # mec_qは, システム内のMECのリスト
        for i in range(self.MEC_NUM):
            self.mec_q.append(Mec(i+1, 0, 0, 300)) # MECをMEC_NUMだけ生成し, mec_qにappendしていく
        self.vehicle = self.vehicle_q[0] # 現在のstepで処理対象になっている車両インスタンスを紐付ける (クリアなコードにするためにこのように実装. 本質的には必要ない.)
        self.mec = self.searchNearestMec(self.vehicle) # オフロード先のMECインスタンスを紐付ける
        self.score_now = 0 # 現在のタイムスロットでのscore
        self.score_prev = 0 # 1つ前のタイムスロットでのscore
             #score = (処理に成功した高重要度タスク数 * IMPORTANCE_RATIO + 処理に成功した低重要度タスク数) / (発生した高重要度タスク数 * IMPORTANCE_RATIO + 発生した低重要度タスク数)
             #score_nowは現在のタイムスロットの評価を表す. どれだけタスクの処理に成功したか, の割合を意味する. -> 長期的報酬の計算に使用
        self.process_time_now = 0 # 現在のタイムスロットで, タスク処理にかかった時間の合計 (現時点では未使用, 長期的報酬の計算などに使用する可能性を考慮し, 実装を残しておく)
        self.process_time_prev = 0 # 1つ前のタイムスロットで, タスク処理にかかった時間の合計 (現時点では未使用, 長期的報酬の計算などに使用する可能性を考慮し, 実装を残しておく)
        self.energy_consumption_now = 0 # 1つ前のタイムスロットでのエネルギー消費の合計 (現時点では未使用, 長期的報酬の計算などに使用する可能性を考慮し, 実装を残しておく)
        self.energy_consumption_prev = 0 # 現在のタイムスロットでのエネルギー消費の合計 (現時点では未使用, 長期的報酬の計算などに使用する可能性を考慮し, 実装を残しておく)

        self.max_now = 0 # score_nowの計算に使用
        
        self.start_flag = False # システム内の時間が終了時刻に達したか, のbool値. 初期化としてFalseに設定.

        # replay bufferへの書き込みindexを保持する変数
        self.previous_pos = 0

        # 交通流データのロード
        self.loadVehicleTrajectory()
        # 交通流データのセット
        self.setVehicleTrajectory()
        # 時刻1における車両の位置をセット
        self.setVehicleLocation(self.time)

        # 評価のための変数の初期化
        self.violate_counter = 0 # 計算タスクがタスクキューに入らなかった回数
        self.violate_counter_sc = 0 # 安全に関わるタスク(safe critical)において, 計算タスクがタスクキューに入らなかった回数
        self.fail_counter = 0 # 遅延制約内での処理に失敗した計算タスクの数
        self.fail_counter_sc = 0 # 安全に関わるタスク(safe critical)において, 遅延制約内での処理に失敗した計算タスクの数
        self.success_counter = 0 #遅延制約内での処理に成功した計算タスクの数
        self.success_counter_sc = 0 # 安全に関わるタスク(safe critical)において, 遅延制約内での処理に成功した計算タスクの数
        self.total_process_time = 0 # タスクの処理にかかった時間の合計
        self.total_process_time_sc = 0 # 安全に関わるタスク(safe critical)において, タスクの処理にかかった時間の合計
        self.total_utility = 0 # システム効用の合計
        self.total_utility_sc = 0 # システム効用の合計 (safe criticalなタスクの処理)
        self.evaluation_data = [] # 評価のためのリスト. i番目要素は重要度iのタスクに関する情報(タスク数, 成功タスク数, 処理時間の合計)を表す.
        self.evaluation_data_wl = [] # 学習中の安全タスク成功率の評価のためのリスト. i番目要素は, タイムスロット(i-1)*100〜i*100までの安全タスクに関する情報(発生した安全タスク数, 成功数)を表す.
        for i in range(10):
            self.evaluation_data.append([0,0,0])
        for i in range(TERMINATE_TIME//SAFETY_EVALUATION_TIMESTEP): # 100秒ごとに記録する実装にする.
            self.evaluation_data_wl.append([0,0])
        

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)

        # 終了時刻に達したあとにresetが呼ばれた場合は, カウンタ系を含めた全ての変数を初期化する必要がある.
        if self.start_flag == True:
            self.violate_counter = 0
            self.violate_counter_sc = 0
            self.fail_counter = 0
            self.fail_counter_sc = 0
            self.success_counter = 0
            self.success_counter_sc = 0
            self.total_process_time = 0
            self.total_process_time_sc = 0
            self.time = 1
            self.vehicle_q.clear()
            for i in range(self.VEHICLE_NUM):
                self.vehicle_q.append(Vehicle(i+1))
            self.vehicle_trajectory = []
            self.mec_q.clear()
            for i in range(self.MEC_NUM):
                self.mec_q.append(Mec(i+1, 0, 0, 300))
            self.time = 1
            self.score_now = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.score_prev = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.total_utility = 0
            self.total_utility_sc = 0
            self.process_time_now = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.process_time_prev = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.energy_consumption_prev = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.energy_consumption_now = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.start_flag = False
            self.loadVehicleTrajectory()
            self.setVehicleTrajectory()
            self.evaluation_data = [] # 評価のためのリスト. i番目要素は重要度iのタスクに関する情報(タスク数, 成功タスク数, 処理時間の合計)を表す.
            self.evaluation_data_wl = [] # 学習中の安全タスク成功率の評価のためのリスト. i番目要素は, タイムスロット(i-1)*100〜i*100までの安全タスクに関する情報(発生した安全タスク数, 成功数)を表す.
            for i in range(10):
                self.evaluation_data.append([0,0,0])
            for i in range(TERMINATE_TIME//SAFETY_EVALUATION_TIMESTEP):
                self.evaluation_data_wl.append([0,0])

        # 1タイムスロットごとにresetする変数.
        task_create = False
        while task_create == False:
            task_create = self.createTaskForVehicle()
        self.mec = self.searchNearestMec(self.vehicle) # オフロード先のMECインスタンスを紐付ける (後ほどstep内で最適なMECに更新. 本質的には不要.)
        self.setVehicleLocation(self.time) # 次の時刻の車両の位置をセット
        self.processed_vehicle_num = 0 # 現在のタイムスロットでオフロード戦略の決定が完了した車両数(processed_vehicle_num)を0に
        self.max_now = 0 # score_nowの計算に使用
        self.local_tq_caltime = self.calTotalsize(self.vehicle.local_tq) / self.F_VEHICLE # 観測で用いるlocalキューに並ぶタスクの処理時間を, 車両v0のlocalタスクキューの値に.
        self.mec_tq_caltime = self.calTotalsize(self.mec.mec_tq) / self.F_MEC # 観測で用いるMECキューに並ぶタスクの処理時間を計算. (注意：RSUとの通信時間は含まれていない)

        # reset() は観測とinfoを返す
        return np.array([self.local_tq_caltime, self.mec_tq_caltime, self.vehicle.task.compute_size, self.vehicle.task.max_delay, self.vehicle.task.importance, self.vehicle.task.is_safe_critical]).astype(np.float32), {}  # infoは空白のdictとする


    # 交通流データ(.csv)をロードする
    def loadVehicleTrajectory(self):
        for i in range(self.VEHICLE_NUM):
            file_name = "vehicle_" + str(i) + ".csv"
            with open(file_name) as f:
                reader = csv.reader(f)
                l = [row for row in reader]
                self.vehicle_trajectory.append(l) # 車両iの交通流データをappend. 最終的に, i番目の要素(=List)が車両iの交通流を表すことになる.
                # 注意：i番目の要素(=車両iの交通流データ)のj番目要素が, 車両iの時刻jにおける座標情報[x,y]になっている
    
    # 車両インスタンスに対し, 交通流データをセットする
    def setVehicleTrajectory(self):
        for i in range(self.VEHICLE_NUM):
            self.vehicle_q[i].trajectory = self.vehicle_trajectory[i]
    
    # 車両インスタンスに対し, 時刻tでの位置をセットする
    def setVehicleLocation(self, t):
        for i in range(self.VEHICLE_NUM):
            self.vehicle_q[i].setLocation(t)

    # タイムスロットを跨いだときに, 各車両に対して計算タスクを生成
    def createTaskForVehicle(self):
        for v in self.vehicle_q: # vehicle_q 内の各車両に対してタスクを生成
            v.createTask(self.ARRIVAL_RATE)
        # self.vehicleをセット. 1回目のstep()で処理される車両を指すようにセットする.
        i = 0
        while self.vehicle_q[i].exist_task == False:
            if i == self.VEHICLE_NUM - 1: # 1つの車両もKFがを持っていない場合
                return False
            i += 1
        self.vehicle = self.vehicle_q[i]
        return True

    # オフロード戦略(action)をとったときの, エネルギー消費の値を計算
    def calEnergyComsumption(self, action):
        if (action == self.LOCAL):
            cal_time = self.vehicle.task.compute_size / self.F_VEHICLE 
            return cal_time * (self.F_VEHICLE ** 2)
        else:
            cal_time = self.vehicle.task.compute_size / self.F_MEC
            return cal_time * (self.F_MEC ** 2)


    # タスクの処理時間とエネルギー消費をジョイントした, Utilityを計算する関数
    def calUtility(self, action):
        process_time = self.calProcessTime(action)
        energy_comsumption = self.calEnergyComsumption(action)
        #utility = (self.ALPHA * (self.REWARD + (self.vehicle.task.max_delay - process_time)) * self.vehicle.task.importance) - (1 - self.ALPHA) * energy_comsumption
        #utility = self.REWARD * self.vehicle.task.importance
        utility = (self.REWARD + (self.vehicle.task.max_delay - process_time) / self.vehicle.task.max_delay) * self.vehicle.task.importance
        #print("vehicle : ", self.vehicle.id, " / task_max_delay : ", self.vehicle.task.max_delay, " / process_time : ", process_time)
        """
        if self.vehicle.task.importance == 1:
            utility = (self.ALPHA * (self.REWARD + (self.vehicle.task.max_delay - process_time)) * self.IMPORTANCE_RATIO) - (1 - self.ALPHA) * energy_comsumption
        else:
            utility = (self.ALPHA * (self.REWARD + (self.vehicle.task.max_delay - process_time)) * 1) - (1 - self.ALPHA) * energy_comsumption
        """
        return utility


    # タスクの処理にかかる時間(=process_time)を計算
    # def appendTask(self, action, vehicle_no) によってタスクキューに計算タスクをappendしてから使うことに注意.
    def calProcessTime(self, action):
        # actionがローカルでの処理の場合 -> タスクの計算にかかる時間のみ
        if action == self.LOCAL:
            # self.vehicleは現在のstepで処理対象となっている車両のNoを表す. 処理対象となっている車両のローカルタスクキューを計算に用いる.
            calculate_time = self.calTotalsize(self.vehicle.local_tq) / self.F_VEHICLE
            process_time = calculate_time
        # actionがMECへのオフロードである場合 -> タスクの計算にかかる時間 + タスクの入力データの送信にかかる時間
        else:
            calculate_time = self.calTotalsize(self.mec.mec_tq) / self.F_MEC
            communicate_time = self.calCommTime(self.vehicle, self.mec)
            process_time = calculate_time + communicate_time
        return process_time


    # def calProcessTime~~ 内で用いる関数. タスクキューに並ぶ計算タスクの総計算量を計算.
    def calTotalsize(self, tq): # tq (task queue) は local_tq か mec_tq のいずれか
        total_size = 0
        for t in tq:
            total_size = total_size + t.compute_size
        return total_size
    
    def isCommunicatable(self, vehicle, mec): # 通信対象のMECと車両を引数として取り, 通信可能であればTrue, 不可能ならFalseを返す関数
        l = math.sqrt((vehicle.x - mec.x) ** 2 + (vehicle.y - mec.y) ** 2) # 通信距離を算出
        if l <= mec.range: # MECの通信可能範囲に車両が存在している場合
            return True
        else: # MECの通信可能範囲に車両が存在していない場合
            return False
    
    def searchNearestMec(self, vehicle): # 車両に対して最も近いMECを探し, そのMECインスタンスを返す関数
        min_l = 100000000 # 十分に大きい値を設定
        nearest_mec = self.mec_q[0] # ID 1 のMECを初期値として設定
        for mec in self.mec_q:
            l = math.sqrt((vehicle.x - mec.x) ** 2 + (vehicle.y - mec.y) ** 2) # 通信距離を算出
            if l < min_l: # 距離lの最小値を下回る距離であれば更新
                min_l = l
                nearest_mec = mec
        return nearest_mec # 最小距離にあるMECをreturn
    
    def calCommRate(self, vehicle, mec): # IEEE802.11pを想定し, そのフィールドテストの結果から伝送レートを算出.
        l = math.sqrt((vehicle.x - mec.x) ** 2 + (vehicle.y - mec.y) ** 2) # 通信距離を算出
        if l < 280: # 280m以内なら, 8Mbpsで固定
            return 8
        elif 280 <= l < 400: # 280m〜400mでは伝送レートは8Mbpsから0Mbpsまで線形に低下
            grad = (0 - 8) / (400 - 280)
            return 8 + grad * (l - 280)

    def calCommTime(self, vehicle, mec): # 通信対象のMECと車両を引数として取り, 通信時間を返す関数
        input_size = vehicle.task.input_size # 車両が持つ計算タスクの入力データ量
        rate = self.calCommRate(vehicle, mec) # IEEE802.11pを想定し, そのフィールドテストの結果から伝送レートを算出.
        return input_size / rate # 計算タスクの入力データ量を通信レートで割り, 通信時間を算出

    # 評価のためのリストを操作する関数.
    def updateEvaluationData(self, task, success, process_time):
        if task.is_safe_critical == 0: # 安全に関わらないタスク
            importance = int(task.importance * 10)
            self.evaluation_data[importance][0] += 1
            self.evaluation_data[importance][2] += process_time
            if success == True: # タスクの処理に成功
                self.evaluation_data[importance][1] += 1
        else: # 安全に関わるタスク
            self.evaluation_data_wl[(self.time - 1)//SAFETY_EVALUATION_TIMESTEP][0] += 1
            self.total_process_time_sc += process_time
            if success == True: # タスクの処理に成功
                self.evaluation_data_wl[(self.time - 1)//SAFETY_EVALUATION_TIMESTEP][1] += 1

        # 100秒ごとに安全に関わるタスクに関する記録を出力してみる (仮実装, 最終的には消去予定)
        #if (self.time % 100 == 0) and (self.vehicle.id == self.VEHICLE_NUM) and (self.is_training == True):
            #print("safe-critical success rate (while learning 〜", self.time, ") : ", round(((self.evaluation_data_wl[(self.time - 1)//100][1] / self.evaluation_data_wl[(self.time - 1)//100][0]) * 100), 1), "%")
        return

    # タスクキューに計算タスクを追加する関数. violateが起こったかどうかを返り値とする.
    def appendTask(self, action):
        violation = False

        if action == self.LOCAL:
            # 以下のif文が成立する場合, local_tqは最大長になってしまっていて, 新たにタスクをappendできない -> violate判定
            if len(self.vehicle.local_tq) >= self.LOCAL_CAPACITY:
                violation = True
                # violateが発生した場合にはfail_counterを増やす (タスクの処理としては失敗と判定)
                if self.vehicle.task.is_safe_critical == 0: # 安全に関わらないタスク
                    self.fail_counter = self.fail_counter + 1
                    self.violate_counter = self.violate_counter + 1
                else: # 安全に関わるタスク
                    self.fail_counter_sc = self.fail_counter_sc + 1
                    self.violate_counter_sc = self.violate_counter_sc + 1
            # violateが起きていない (=タスクキューが最大長ではない) 場合は, local_tqにタスクをappendする.
            else:
                self.vehicle.local_tq.append(self.vehicle.task)

        elif action == self.MEC:
            # 以下のif文が成立する場合, mec_tqは最大長になってしまっていて, 新たにタスクをappendできない -> violate判定
            if len(self.mec.mec_tq) >= self.MEC_CAPACITY:
                violation = True
                # violateが発生した場合にはfail_counterを増やす (タスクの処理としては失敗と判定)
                self.fail_counter = self.fail_counter + 1
                self.violate_counter = self.violate_counter + 1
            # violateが起きていない (=タスクキューが最大長ではない) 場合は, mec.mec_tqにタスクをappendする.
            else:
                self.mec.mec_tq.append(self.vehicle.task)
        #violateが起こったかどうかを返り値とする. -> 報酬を決定するときに使用
        return violation


    # タスクキュー内の計算タスクを処理する関数. タイムスロットの終わりに呼び出して実行. 1秒を1タイムスロットとしているので, 1秒分のタスク処理を進める.
    def processTask(self):
        # local_tq 内の計算タスクを処理
        for i in range(self.VEHICLE_NUM): # 全車両についてlocal_tqを1秒ぶん処理し, 更新
            time = 0 # time(=処理時間の合計)が1になるまで処理
            temp_time = 0 # 処理の終了判定に使用
            stop = False # 時間が1タイムスロット(=1秒)を越えるかどうかを意味するbool値
            while (stop == False) and (len(self.vehicle_q[i].local_tq) != 0):
                now_processed = self.vehicle_q[i].local_tq[0] # 現在処理対象としている計算タスクを指す. タスクキュー左端に最も時系列的に古い計算タスクが存在しているため, 左端から処理していく.
                temp_time = time + (now_processed.compute_size / self.F_VEHICLE) # 現在対象の計算タスクを終わりまで処理したとして, timeがどうなるか, という仮の値を計算
                if temp_time <= 1: # 現在対象の計算タスクを完全に処理してもまだ処理時間に余裕があるなら, 現在対象の計算タスクは処理完了とする.
                    time = temp_time # 処理時間の合計を更新
                    # 注意 : タスクキュー内, 左端に最も時系列的に古い計算タスクが存在しているため, 左端からpopしていく必要がある.
                    self.vehicle_q[i].local_tq.popleft()
                else: # 現在対象の計算タスクを完全に処理するだけの残り時間がなければ, できるところまで処理して次のタイムスロットへ移行することになる.
                    now_processed.compute_size = now_processed.compute_size - (1 - time) * self.F_VEHICLE # compute_sizeを更新. 残り時間でできる分だけ処理されるため減算.
                    stop = True # 処理の終了判定

        # mec_tq 内の計算タスクを処理
        for i in range(self.MEC_NUM): # 全MECについてmec_tqを1秒ぶん処理し, 更新
            time = 0 # time(=処理時間の合計)が1になるまで処理
            temp_time = 0 # 処理の終了判定に使用
            stop = False # 時間が1タイムスロット(=1秒)を越えるかどうかを意味するbool値
            while (stop == False) and (len(self.mec_q[i].mec_tq) != 0):
                now_processed = self.mec_q[i].mec_tq[0] # 現在処理対象としている計算タスクを指す. タスクキュー左端に最も時系列的に古い計算タスクが存在しているため, 左端から処理していく.
                temp_time = time + (now_processed.compute_size / self.F_MEC) # 現在対象の計算タスクを終わりまで処理したとして, timeがどうなるか, という仮の値を計算
                if temp_time <= 1: # 現在対象の計算タスクを完全に処理してもまだ処理時間に余裕があるなら, 現在対象の計算タスクは処理完了とする.
                    time = temp_time # 処理時間の合計を更新
                    # 注意 : タスクキュー内, 左端に最も時系列的に古い計算タスクが存在しているため, 左端からpopしていく必要がある.
                    self.mec_q[i].mec_tq.popleft()
                else: # 現在対象の計算タスクを完全に処理するだけの残り時間がなければ, できるところまで処理して次のタイムスロットへ移行することになる.
                    now_processed.compute_size = now_processed.compute_size - (1 - time) * self.F_MEC # compute_sizeを更新. 残り時間でできる分だけ処理されるため減算.
                    stop = True # 処理の終了判定


    # 各種, ~~_prevの値を更新する関数
    def updatePrev(self):
        self.process_time_prev = self.process_time_now
        self.energy_consumption_prev = self.energy_consumption_now
        self.score_prev = self.score_now
        return


    def _update_safety_model(self):
        """
        SafetyUncertaintyModel を更新する関数.
        一定数のデータがバッファに溜まったら呼び出す.
        """
        if len(self.safety_replay_buffer) < 256: # バッチサイズ分のデータが溜まるまで待つ
            return
        
        # バッファからランダムにバッチを取得
        batch_size = 256
        batch = random.sample(self.safety_replay_buffer, batch_size)
        states, actions, actual_safety_costs = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        actual_safety_costs = torch.tensor(np.array(actual_safety_costs), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        self.safety_optimizer.zero_grad()
        mu_pred, var_pred = self.safety_uncertainty_model(states, actions)
        
        # ガウス分布の負の対数尤度を損失とする (負のログ尤度を最小化)
        log_prob = -0.5 * torch.log(2 * torch.pi * var_pred) - 0.5 * ((actual_safety_costs - mu_pred) ** 2) / var_pred
        loss = -log_prob.mean()
        
        # あるいは、MSE + 分散のペナルティ (不確実性を奨励しない)
        # loss = torch.mean((mu_pred - actual_safety_costs) ** 2) + torch.mean(var_pred) 
        
        loss.backward()
        self.safety_optimizer.step()

    def _calculate_safety_uncertainty(self, obs, action):
        """
        観測とアクションから安全コストの予測と不確実性 (Gamma) を計算する
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu_pred, var_pred = self.safety_uncertainty_model(obs_tensor, action_tensor)
        
        # Gamma は分散の平方根, または分散そのものにスケールをかけたものなど, 定義による
        # 論文の定義 (mu +/- Gamma) から, Gamma は標準偏差に近い
        gamma = torch.sqrt(var_pred).squeeze().item() 
        mu = mu_pred.squeeze().item()
        
        return mu, gamma

    def _determine_safe_actions(self, current_obs):
        """
        現在の観測に基づいて、安全と予測されるアクションの集合を返す
        """
        possible_actions = [self.LOCAL, self.MEC]
        safe_actions = []
        
        for a in possible_actions:
            # _calculate_safety_uncertainty はスカラー値を返すように修正
            predicted_mu, predicted_gamma = self._calculate_safety_uncertainty(current_obs, a)
            #print(current_obs)
            #print(predicted_mu, " / ", predicted_gamma)
            
            # 安全制約条件: mu + Gamma <= b_h
            # current_obs[5] は task.is_safe_critical に対応
            task_is_safe_critical = current_obs[5] 
            
            if task_is_safe_critical == 1: # 安全に関わるタスク
                # 安全コストの予測 (predicted_mu) が閾値 (self.safety_threshold_b_h) を超えない
                # かつ、不確実性 (predicted_gamma) を考慮した信頼上限が閾値を超えない
                if predicted_mu + predicted_gamma <= self.safety_threshold_b_h:
                    safe_actions.append(a)
            else: # 低重要度タスクは常に安全とみなす（または別の緩い制約を適用）
                safe_actions.append(a)
                
        return safe_actions

    # 安全なアクションが存在しないときに緊急アクションをとる
    def takeEmergencyAction(self):
        epsilon = 1e-6  # 浮動小数点誤差の許容範囲
        loss_importance_local = 0 # 緊急アクションをlocal_tqに適用した場合に損失するタスクの重要度の合計
        loss_importance_mec = 0 # 緊急アクションをmec_tqに適用した場合に損失するタスクの重要度の合計

        # タスクキューを複製
        local_tq_copy = copy.deepcopy(self.vehicle.local_tq)
        mec_tq_copy = copy.deepcopy(self.mec.mec_tq)

        l_sc = True
        m_sc = True

        # local_tqについて, 重要度が低いタスクから順に削除し, 安全に関わるタスクが遅延制約を満たすように処理
        while ((self.calTotalsize(local_tq_copy) + self.vehicle.task.compute_size) / self.F_VEHICLE) > self.vehicle.task.max_delay:
            # safe criticalではないタスクのみを削除対象とする
            removable_tasks = [task for task in local_tq_copy if task.is_safe_critical == 0]
            if not removable_tasks:
                l_sc = False
                break

            min_importance = min(task.importance for task in removable_tasks)

            # 削除前のリストの長さを取得
            before_len = len(local_tq_copy)

            # 削除（isclose で比較）
            local_tq_copy = [
                task for task in local_tq_copy
                if not (math.isclose(task.importance, min_importance, abs_tol=epsilon) and task.is_safe_critical == 0)
            ]

            # 削除後のリストの長さを取得
            after_len = len(local_tq_copy)

            # 削除前後のリストの長さの差から, 損失した重要度を計算 -> loss_importance に加算
            loss_importance_local = loss_importance_local + (before_len - after_len) * min_importance

        # mec_tqについて, 重要度が低いタスクから順に削除し, 安全に関わるタスクが遅延制約を満たすように処理
        while ((self.calTotalsize(mec_tq_copy) + self.vehicle.task.compute_size) / self.F_MEC) > self.vehicle.task.max_delay:
            removable_tasks = [task for task in mec_tq_copy if task.is_safe_critical == 0]
            if not removable_tasks:
                m_sc = False
                break

            min_importance = min(task.importance for task in removable_tasks)

            # 削除前のリストの長さを取得
            before_len = len(mec_tq_copy)

            mec_tq_copy = [
                task for task in mec_tq_copy
                if not (math.isclose(task.importance, min_importance, abs_tol=epsilon) and task.is_safe_critical == 0)
            ]

            # 削除後のリストの長さを取得
            after_len = len(mec_tq_copy)

            # 削除前後のリストの長さの差から, 損失した重要度を計算 -> loss_importance に加算
            loss_importance_mec = loss_importance_mec + (before_len - after_len) * min_importance
        
        #print("l_sc : ", l_sc, " / m_sc : ", m_sc)

        # local_tqに緊急アクションを適用した場合とmec_tqに緊急アクションを適用した場合を比較し, より重要度の損失が小さい方を採用
        if ((loss_importance_local <= loss_importance_mec) or (m_sc == False)) and (l_sc == True):
            # 長期的報酬のための変数(self.score_now)と, 評価のためのデータを更新
            set1 = set(self.vehicle.local_tq)
            set2 = set(deque(local_tq_copy))
            diff = list(set1 - set2)
            for task in diff:
                importance = int(task.importance * 10)
                self.evaluation_data[importance][1] -= 1
                self.success_counter -= 1
                self.fail_counter += 1
                self.score_now -= self.REWARD * task.importance
                self.total_utility -= task.utility
            self.vehicle.local_tq = deque(local_tq_copy)
            self.vehicle.local_tq.append(self.vehicle.task)
            process_time = self.calTotalsize(self.vehicle.local_tq) / self.F_VEHICLE
            
            return True, process_time
        
        elif ((loss_importance_local > loss_importance_mec) or (l_sc == False)) and (m_sc == True):
            # 長期的報酬のための変数(self.score_now)と, 評価のためのデータを更新
            set1 = set(self.mec.mec_tq)
            set2 = set(deque(mec_tq_copy))
            diff = list(set1 - set2)
            for task in diff:
                importance = int(task.importance * 10)
                self.evaluation_data[importance][1] -= 1
                self.success_counter -= 1
                self.fail_counter += 1
                self.score_now -= self.REWARD * task.importance
            self.mec.mec_tq = deque(mec_tq_copy)
            self.mec.mec_tq.append(self.vehicle.task)
            process_time = self.calTotalsize(self.mec.mec_tq) / self.F_MEC

            return True, process_time
        
        else:
            return False

    # 観測とアクションのペアに対し, 安全コストg(s,a)を計算
    def calSafetyCost(self, observation, action):
        # 観測は, (localでの処理にかかる処理時間, mecへのオフロードでかかる処理時間, 計算タスクの計算量, 計算タスクの遅延制約, 計算タスクの重要度, 安全に関わるタスクかどうか)
        local_cal_time = observation[0]
        mec_cal_time = observation[1]
        task_compute_size = observation[2]
        task_max_delay = observation[3]
        task_importance = observation[4]
        task_is_safe_critical = observation[5]
        
        if action == self.LOCAL: # actionがローカルでの処理だった場合
            process_time = local_cal_time + self.vehicle.task.compute_size / self.F_VEHICLE # 処理時間 = (観測から得た)計算時間
        else: # actionがMECへのオフロードだった場合
            process_time = mec_cal_time + self.vehicle.task.compute_size / self.F_MEC + self.calCommTime(self.vehicle, self.mec) # 処理時間 = (観測から得た)計算時間 + 通信にかかる時間
        
        if process_time <= self.vehicle.task.max_delay: # 遅延制約内での処理が可能な場合
            safety_cost = 0
        else: # 遅延制約内での処理に失敗する場合
            safety_cost = 1 # 安全コストが高い = 危険であることを表す
        
        return safety_cost

    def step(self, action):

        #print(self.model.replay_buffer.rewards[self.previous_pos])

        # 長期的報酬の計算のために, max_nowの値を更新
        if self.vehicle.task.is_safe_critical == 0: # 計算タスクが安全に関わらないタスクであれば更新
            self.max_now = self.max_now + self.REWARD * self.vehicle.task.importance
        
        # actionが不正な値だった場合にはerror. 通常起こらないはず.
        if (action != self.LOCAL) and (action != self.MEC):
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        
        # 現在の観測を保存
        current_obs = np.array([self.local_tq_caltime, self.mec_tq_caltime, 
                                self.vehicle.task.compute_size, self.vehicle.task.max_delay, self.vehicle.task.importance, self.vehicle.task.is_safe_critical]).astype(np.float32)

        pass_normal_processing = False

        # MASE: 安全なアクションの決定
        # 高重要度タスクの場合のみ、安全なアクションかどうかをチェックし, 緊急アクションへの変更やsheldingによる安全アクションへの変更を行う.
        if self.vehicle.task.is_safe_critical == 1:
            # 論文の Algorithm 1 の Line 3 に相当: Take "safe" action a_h=pi(s_h) within A_h^+
            safe_actions = self._determine_safe_actions(current_obs)

            # 論文の Algorithm 1 の Line 6-9 に相当: Emergency Stop Action
            if not safe_actions: # safe_actions が空の場合 (実行可能な安全アクションがない)
                # 緊急停止アクションを実行
                penalty = - (self.PENALTY_SCALAR_C / self._calculate_safety_uncertainty(current_obs, action)[1])
                reward = 0
                #print("(emergency) penalty reward = ", reward)
                # ここでのペナルティ計算は論文の式(1)に基づく。
                # min_{a in A} Gamma(s_{h+1}, a) を取得する必要があるが、簡略化のため current_obs の不確実性を使う
                # または、非常に大きな固定ペナルティを設定する
                # reward = -PENALTY_SCALAR_C # 固定ペナルティの場合

                success, process_time = self.takeEmergencyAction() # 緊急アクションの実行
                if self.model is not None:
                    self.model.replay_buffer.rewards[self.previous_pos] = np.array(penalty) # replay bufferの書き換え

                if success == True: # 緊急アクションの結果, 安全に関わるタスクの処理に成功した場合
                    self.success_counter_sc = self.success_counter_sc + 1
                    self.updateEvaluationData(self.vehicle.task, True, process_time)
                else: # 緊急アクションの結果, 安全に関わるタスクの処理に失敗した場合
                    self.fail_counter_sc = self.fail_counter_sc + 1
                    self.updateEvaluationData(self.vehicle.task, False, process_time)
                
                self.safety_replay_buffer.append((current_obs, action, self.calSafetyCost(current_obs, action))) # 安全コストを推定するモデルの学習のため, bufferに情報を追加.
                #print(f"Time {self.time}, Vehicle {self.vehicle.id}: EMERGENCY STOP! No safe action.")

                self._update_safety_model()
                pass_normal_processing = True # 通常の処理をスキップ

            # もし、エージェントが推奨するアクション (model.predictから来たaction) が安全でないと判断されたら
            elif action not in safe_actions:
                #print(f"Time {self.time}, Vehicle {self.vehicle.id}: FORCED EMERGENCY STOP! Agent chose unsafe action {action}. Safe actions were {safe_actions}.")
                # shelding として, 安全なアクションに変更
                action = random.choice(safe_actions) # アクションはLOCALかMECのどちらかしかないので, 強制的にoriginalのアクションではない方に変更される.
                #reward = - (self.PENALTY_SCALAR_C / self._calculate_safety_uncertainty(current_obs, action)[1]) # エージェントが選んだunsafe actionの不確実性で割る
                self.safety_replay_buffer.append((current_obs, action, self.calSafetyCost(current_obs, action))) # 安全コストを推定するモデルの学習のため, bufferに情報を追加.
                self._update_safety_model() 

        if pass_normal_processing == False: # 以下, 通常の処理 (緊急アクションが取られていない場合の処理)
            violation = self.appendTask(action)
            if violation != True:
                reward = 0
            else:
                reward = -1 # violateが起こった場合には報酬としてペナルティ

            # violateが起こっていなければタスクの処理時間(process_time)を計算し, タスクの遅延制約内に処理できているか判定 -> 短期的報酬の設定
            if violation != True:
                # タスクの処理にかかる時間(=process_time)を計算
                process_time = self.calProcessTime(action)

                if action == self.MEC: # オフロード戦略がMECへのオフロードであれば, 対象MECのリンク数を1増やす
                    self.mec.link = self.mec.link + 1

                # 評価のため, タスクの処理時間をこれまでの処理時間の合計に加える
                self.total_process_time = self.total_process_time + process_time

                # process_time_now の算出のため, タスクの処理時間をこれまでのprocess_time_nowに加える
                self.process_time_now = self.process_time_now + process_time

                # タスク処理によるエネルギー消費を算出
                energy_comsumption = self.calEnergyComsumption(action)

                # energy_comsumption_now の算出のため, エネルギー消費をこれまでのenergy_comsumption_nowに加える
                self.energy_consumption_now = self.energy_consumption_now + energy_comsumption

                # タスクの遅延制約内にタスクの処理が完了する場合 (=タスクの処理成功の場合)
                if process_time <= self.vehicle.task.max_delay:
                    utility = self.calUtility(action) # 処理成功によるutilityを計算
                    reward = utility # 短期的報酬を設定

                    if self.vehicle.task.is_safe_critical == 0: # 安全に関わらないタスクであれば通常の処理
                        self.score_now = self.score_now + self.REWARD * self.vehicle.task.importance
                        self.success_counter = self.success_counter + 1
                        self.total_utility = self.total_utility + utility # 評価のため, 通算のUtilityを更新
                        self.vehicle.task.utility = utility
                    else: # 安全に関わるタスクであれば, 長期的報酬を算出するためのscore_nowには加えず, 評価のためのカウンタのみを操作
                        self.success_counter_sc = self.success_counter_sc + 1
                        self.total_utility_sc = self.total_utility_sc + utility
                    self.safety_replay_buffer.append((current_obs, action, self.calSafetyCost(current_obs, action))) # 安全コストを推定するモデルの学習のため, bufferに情報を追加.
                    self._update_safety_model()

                    self.updateEvaluationData(self.vehicle.task, True, process_time)

                # タスクの遅延制約内にタスクの処理が完了しない場合 (=タスクの処理失敗の場合)
                else:
                    if self.vehicle.task.is_safe_critical == 0: # 安全に関わらないタスクであれば, 通常の処理
                        self.fail_counter = self.fail_counter + 1
                        reward = - self.REWARD * self.vehicle.task.importance # 短期的報酬として, 固定報酬の負の値をrewardとする
                    else: # 安全に関わるタスクであれば, 非常に大きいペナルティをつける
                        self.fail_counter_sc = self.fail_counter_sc + 1
                        reward = -100
                    self.safety_replay_buffer.append((current_obs, action, self.calSafetyCost(current_obs, action))) # 安全コストを推定するモデルの学習のため, bufferに情報を追加.
                    self._update_safety_model()
                    self.updateEvaluationData(self.vehicle.task, False, process_time)

        self.processed_vehicle_num = self.processed_vehicle_num + 1 # オフロード戦略の決定が完了した車両数を1増やす
        if self.processed_vehicle_num >= self.VEHICLE_NUM: # オフロード戦略の決定が完了した車両数がシステム内の車両数に達した場合, そのタイムスロット(=episode)は終了
            terminated = True
        else: # 現在のタイムスロットが終了していない場合
            terminated = False
            self.vehicle = self.vehicle_q[self.processed_vehicle_num] # 次の処理対象の車両をself.vehicleに設定
            self.mec = self.searchNearestMec(self.vehicle) # 次の処理対象の車両に対し, 最も近いMECをself.mecに設定
        
        # 次の処理対象になる車両に計算タスクが存在していない場合と, 次の処理対象になる車両と通信可能なMECがない場合には特別な対応が必要
        while ((self.vehicle.exist_task == False) or (self.isCommunicatable(self.vehicle, self.mec) == False)):
            if self.vehicle.exist_task == False: # 次の処理対象になる車両に計算タスクが存在していない場合, その車両の処理はskipする.
                self.processed_vehicle_num = self.processed_vehicle_num + 1 # オフロード戦略の決定が完了した車両数を1増やす (車両に計算タスクが存在しなかったため)
            else: # 次の処理対象になる車両と通信可能なMECがない場合, タスクをローカルで処理
                self.appendTask(self.LOCAL) # オフロード戦略は強制的にLOCALとなる.
                self.processed_vehicle_num = self.processed_vehicle_num + 1 # オフロード戦略の決定が完了した車両数を1増やす

            if self.processed_vehicle_num >= self.VEHICLE_NUM: # オフロード戦略の決定が完了した車両数がシステム内の車両数に達した場合, そのタイムスロット(=episode)は終了
                terminated = True
                break
            else:
                terminated = False # 全車両のオフロード戦略が未決定であれば, まだタイムスロット内での処理であるから, terminated は False
                # processed_vehicle_numには, オフロード戦略の決定が完了した車両数が入っているため, vehicle_qのindexとして使用することで, 次の処理対象の車両を示すことができる.
                # 注意：リストのindexは0オリジンである.
                self.vehicle = self.vehicle_q[self.processed_vehicle_num] # 処理対象の車両を改めて設定
                self.mec = self.searchNearestMec(self.vehicle) # 処理対象の車両に最も近いMECを改めて設定

        #terminated = bool(self.processed_vehicle_num >= self.VEHICLE_NUM) # オフロード戦略の決定が完了した車両数がシステム内の車両数に達した場合, そのタイムスロット(=episode)は終了
        truncated = False  # truncated は今回は定めない

        # タイムスロットの終わりであれば, 以下のコードを実行
        if terminated:
            # score_now について, ここまでタスクが成功するごとに, REWARD (* IMPORTANCE_RATIO) を加算しているため, 全ての処理に成功した場合(=scoreの最大値)で除算し, scoreを算出する.
            #self.score_now = self.score_now / (self.num_h_now * self.REWARD * self.IMPORTANCE_RATIO + self.num_l_now * self.REWARD)
            self.score_now = self.score_now / self.max_now
            # ゼロ除算を避けるため, utilityが0の場合は非常に小さい値にすり替える
            if self.score_now == 0:
                self.score_now = 0.001
                print("WARN : score_now == 0 is True at time ", self.time)
            # システム内の時間が1でなければ, 前のタイムスロットが存在 -> 長期的報酬を計算
            if self.time != 1:
                # 長期的報酬をrewardに加算
                reward = reward + ((self.score_now - self.score_prev) / self.score_prev) * self.BETA
                # rewardが非常に大きい値になってしまい, 局所解に陥ることを防ぐため, reward clippingを導入
                if reward > 50.0:
                    reward = 50.0
                elif reward < -50.0:
                    reward = -50.0
            # 各種prevを更新
            self.updatePrev()
            self.processTask()

            # 100秒ごとに安全に関わるタスクに関する記録を出力してみる (仮実装, 最終的には消去予定)
            if (self.time % SAFETY_EVALUATION_TIMESTEP == 0) and (self.vehicle.id == self.VEHICLE_NUM) and (self.is_training == True):
                with open('success_rate_wl_safeRL.csv', 'a') as f:
                    writer = csv.writer(f)
                    if (self.evaluation_data_wl[(self.time - 1)//SAFETY_EVALUATION_TIMESTEP][0] != 0):
                        writer.writerow([round(((self.evaluation_data_wl[(self.time - 1)//SAFETY_EVALUATION_TIMESTEP][1] / self.evaluation_data_wl[(self.time - 1)//SAFETY_EVALUATION_TIMESTEP][0]) * 100), 1)])
                    else:
                        writer.writerow([0])
                        print("No safe-critical tasks occurred. at time : ", self.time)
                #print("safe-critical success rate (while learning 〜", self.time, ") : ", round(((self.evaluation_data_wl[(self.time - 1)//100][1] / self.evaluation_data_wl[(self.time - 1)//100][0]) * 100), 1), "%")

            self.time = self.time + 1 # システム内の時間を進める

            # 注意：次のタイムスロット(=episode)の初期観測は, reset()によって与えられるため, ここでは観測の情報を更新しない.

            if self.time > self.TERMINATE_TIME: # 終了時刻に達していれば, start_flagをTrueにしてからresetが呼ばれるようにする.
                self.start_flag = True
            else:
              self.start_flag = False # 念の為, 明示的にコーディング

        # タイムスロットの終わりではないなら, 次の車両の処理へ移行
        else:
            # self.vehicleを更新してからcalProcessTimeを呼ぶことで, 次のstepで処理対象となる車両のlocal_tqの情報を, 観測として与える
            self.local_tq_caltime = self.calTotalsize(self.vehicle.local_tq) / self.F_VEHICLE # 観測で用いるlocalキューに並ぶタスクの処理時間を, 車両v0のlocalタスクキューの値に.
            self.mec_tq_caltime = self.calTotalsize(self.mec.mec_tq) / self.F_MEC # 観測で用いるMECキューに並ぶタスクの処理時間を計算. (注意：RSUとの通信時間は含まれていない)

        # infoにはタスクIDを入れておく. replay bufferのinfosに記録され, safeRLの緊急アクション後のreplay buffer書き換えに使用.
        info = {'task_id':self.vehicle.task.id}

        # replay bufferのindexを表すposを更新. 1つ前の経験のindexを指すようにする.
        if self.model is not None:
            self.previous_pos = self.model.replay_buffer.pos

        #print("reward : ", reward)

        return (
            np.array([self.local_tq_caltime, self.mec_tq_caltime, self.vehicle.task.compute_size, self.vehicle.task.max_delay, self.vehicle.task.importance, self.vehicle.task.is_safe_critical]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass

"""###評価の実行"""

base_env = SimpleOffloadEnv()

n_eva = 1
n_learn = N_LEARN

for n in range(n_eva):
  # PPOによる学習 (ent_coefを大きめにとり, 局所解に陥ることを防止)
  random.seed()
  #model = PPO("MlpPolicy", vec_env, ent_coef=0.05, verbose=0)
  #model = DQN("MlpPolicy", vec_env, exploration_initial_eps=1.0, exploration_final_eps=0.1, exploration_fraction=0.5, verbose=0)
  model = DQN("MlpPolicy", base_env, verbose=0) # 一旦はbase_envを対応させてモデルを定義
  base_env = SimpleOffloadEnv(model=model, is_training=True) # 強化学習モデルを紐付けて, 改めて環境を生成. trainingプロセスであることも提示.
  #rnd_env = RNDWrapper(base_env) # RNDによる探索範囲の拡大
  #vec_env = DummyVecEnv([lambda: rnd_env]) # 環境のベクトル化
  vec_env = DummyVecEnv([lambda: base_env])
  model.set_env(vec_env) # 環境を強化学習モデルにセット
  model.learn(total_timesteps=n_learn)
  #model.save(f"/tmp/gym/PPO") # 学習済のPPOモデルを保存
  model.save(f"/tmp/gym/DQN") # 学習済のDQNモデルを保存

  # 以下, 評価のためのコード
  random.seed(100)
  n_steps = VEHICLE_NUM * TERMINATE_TIME
  total_reward = 0
  #env = SimpleOffloadEnvForEvaluation()
  model = DQN.load(f"/tmp/gym/DQN") # 学習済のDQNモデルをロード
  env = SimpleOffloadEnv(model=model)
  #model = PPO.load(f"/tmp/gym/PPO") # 学習済のPPOモデルをロード
  time = 1
  obs, info = env.reset()

  for step in range(n_steps):
      action, _ = model.predict(obs, deterministic=True)
      """
      print("replay buffer ::")
      print("action list : ", model.replay_buffer.actions)
      print("reward list : ", model.replay_buffer.rewards)
      print("next observations list : ", model.replay_buffer.next_observations)
      #print("Time : ", env.time, " / Task importance : ", env.task.importance, " / action : ", action)
      """
      obs, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      total_reward = total_reward + reward

      if done:
          if env.time > TERMINATE_TIME:
            print("EVALUATION -- ", n+1, "/", n_eva)
            print("violate_counter: ", env.violate_counter)
            print("violate_counter (safe critical) : ", env.violate_counter)
            print("success_counter: ", env.success_counter)
            print("fail_counter: ", env.fail_counter)
            print("success_counter (safe critical) : ", env.success_counter_sc)
            print("fail_counter (safe critical) : ", env.fail_counter_sc)
            #print("ave_reward =", round((total_reward / n_steps), 5))
            print("success rate: ", round(env.success_counter / (env.success_counter + env.fail_counter) * 100, 1), "%")
            print("success rate (safe critical) : ", round(env.success_counter_sc / (env.success_counter_sc + env.fail_counter_sc) * 100, 1), "%")
            print("average process time : ", round(env.total_process_time / (env.success_counter + env.fail_counter - env.violate_counter), 3), " sec")
            print("average process time (safe critical) : ", round(env.total_process_time_sc / (env.success_counter_sc + env.fail_counter_sc - env.violate_counter_sc), 3), " sec")
            print("system utility : ", round(env.total_utility, 3))
            print("system utility (safe critical) : ", round(env.total_utility_sc, 3))
            print("system utility (all) : ", round(env.total_utility + env.total_utility_sc, 3))
            print("")
            for i in range(10):
                print("** IMPORTANCE", i, "~", i+1, " **")
                print("success rate: ", round(env.evaluation_data[i][1] / env.evaluation_data[i][0] * 100, 1), "%")
                print("average process time: ", round(env.evaluation_data[i][2]/env.evaluation_data[i][0], 3), " sec")
            print("")
            print("")
            break
          obs, info = env.reset()


###############################################################################################################################################################
""" Greedyアルゴリズムの評価 1 """

"""
def generateGreedyStrategy(data, r, i):
    length = len(data)
    if (i >= int(pow(length, r))):
        print("ERROR: the value of i is too large.")
        return
    element = []
    for digit in reversed(range(r)):
        element.append(data[int(i / pow(length, digit)) % length])
    return element


def greedyAction(env, n):
    best_strategy = []
    estimated_reward_max = -100000000000000

    for a in range(2**n):
        strategy = generateGreedyStrategy([0,1], n, a)
        env_copy = copy.deepcopy(env)
        estimated_reward = 0

        for b in range(n):
            action = strategy[b]
            _, reward, terminated, truncated, _ = env_copy.step(action)
            done = terminated or truncated
            estimated_reward = estimated_reward + reward
            if done:
                break

        if (estimated_reward > estimated_reward_max):
            estimated_reward_max = estimated_reward
            best_strategy = strategy
            #print("--  best strategy updated  --")

    return best_strategy[0]

random.seed()
random.seed(100)
#env = SimpleOffloadEnvForGreedy()
env = SimpleOffloadEnv()
obs, info = env.reset()
n_steps = TERMINATE_TIME * VEHICLE_NUM
total_reward = 0
time = 1

for step in range(n_steps):
    action = greedyAction(env, 1)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward = total_reward + reward
    #print("greedy step: ", step)

    if done:
          if env.time > TERMINATE_TIME:
            print("-- Evaluation for greedy algorithm 1 --")
            #print("violate_counter: ", env.violate_counter_h + env.violate_counter_l)
            print("violate_counter: ", env.violate_counter)
            #print("success_counter_h: ", env.success_counter_h)
            #print("success_counter_l: ", env.success_counter_l)
            print("success_counter: ", env.success_counter)
            #print("fail_counter_h: ", env.fail_counter_h)
            #print("fail_counter_l: ", env.fail_counter_l)
            print("fail_counter: ", env.fail_counter)
            print("success_counter_sc: ", env.success_counter_sc)
            print("fail_counter_sc: ", env.fail_counter_sc)
            print("ave_reward =", round((total_reward / n_steps), 5))
            #print("success rate (high importance) : ", round((env.success_counter_h/env.num_h) * 100, 1), "%")
            #print("success rate (low importance) : ", round((env.success_counter_l/env.num_l) * 100, 1), "%")
            print("success rate: ", round(env.success_counter / (env.success_counter + env.fail_counter) * 100, 1), "%")
            print("success rate (safe critical): ", round(env.success_counter_sc / (env.success_counter_sc + env.fail_counter_sc) * 100, 1), "%")
            #print("average process time : ", round(env.total_process_time/(env.num_h + env.num_l - env.violate_counter_h - env.violate_counter_l), 3), " sec")
            print("average process time : ", round(env.total_process_time/(env.success_counter + env.fail_counter - env.violate_counter), 3), " sec")
            #print("average process time (high importance) : ", round((env.total_process_time_h/(env.success_counter_h + env.fail_counter_h - env.violate_counter_h)), 3), " sec")
            #print("average process time (low importance) : ", round((env.total_process_time_l/(env.success_counter_l + env.fail_counter_l - env.violate_counter_l)), 3), " sec")
            #print("score : ", env.success_counter_h * env.IMPORTANCE_RATIO + env.success_counter_l)
            #print("score : ", env.success_counter_h * env.IMPORTANCE_RATIO + env.success_counter_l)
            print("system utility : ", round(env.total_utility, 3))
            print("")
            for i in range(10):
                print("** IMPORTANCE", i, "~", i+1, " **")
                print("success rate: ", round(env.evaluation_data[i][1] / env.evaluation_data[i][0] * 100, 1), "%")
                print("average process time: ", round(env.evaluation_data[i][2]/env.evaluation_data[i][0], 3), " sec")
            print("")
            print("")
            break
          obs, info = env.reset()
"""
          
""" Greedyアルゴリズムの評価 2 """
"""
class SimpleOffloadEnvForGreedy(gym.Env):

    # オフロード戦略を表す変数を定義
    LOCAL = 0 # タスクが車両のリソースで処理されるオフロード戦略
    MEC = 1 # タスクがオフロードされ, MECのリソースで処理されるオフロード戦略

    ######################################################## 各種パラメータの定義 ##################################################

    ALPHA = 1.0 # 効用関数において, タスクの処理遅延に関する項とエネルギー消費に関する項のバランスを決定するパラメタ.
                # U = ALPHA*(処理遅延に関する項) + (1-ALPHA)*(エネルギー消費に関する項)
    BETA = VEHICLE_NUM * 10 # 長期的報酬と短期的報酬のバランスを決定するパラメタ. 長期的報酬はBETA倍される.
    REWARD = 5 # タスクを遅延制約内で処理成功したときの, 固定効用r
    IMPORTANCE_RATIO = 5 # 高重要度タスクの重要度係数. (低重要度タスクの重要度係数を1としている.)

    # 注意 : 計算タスクの遅延制約や計算量などはclass:taskで定義されている.
    ARRIVAL_RATE = 1 # 車両に対する, 1タイムスロットでの計算タスクの発生確率

    NUM_CORE = 10 # MECのコア数
    F_MEC = 3.2 * NUM_CORE # (G cycles / s) # MECの計算能力
    F_VEHICLE = 0.8 # (G cycles / s) # 車両の計算能力
    VEHICLE_NUM = VEHICLE_NUM # システム内の車両数
    MEC_NUM = 1 # システム内のMEC数
    TERMINATE_TIME = TERMINATE_TIME # シミュレーション終了時間. 1から開始.

    # 通信に関するパラメタの設定
    WAVEGHz = 5.9 # 車両とRSUの無線通信の周波数. IEEE802.11pを仮定.
    BANDWIDTH = 10 # 車両とRSUの無線通信の帯域幅. (MHz)
    A_GAIN_v = 1 # 車両側のアンテナの絶対利得
    A_GAIN_r = 1 # RSU側のアンテナの絶対利得
    T_POWER = 0.1 # 送信電力 (W)
    NOISE = -80 # ノイズの強さ (dBm)

    ##############################################################################################################################


    def __init__(self):

        super(SimpleOffloadEnvForGreedy, self).__init__()

        self.LOCAL_CAPACITY = 200 # local_tqの最大長を定義. (本質的には必要ないが, 観測空間を有限にするために定義. violateが起こらないような十分に長いキュー長にする必要あり.)
        self.MEC_CAPACITY = 200 # mec_tqの最大長を定義. (本質的には必要ないが, 観測空間を有限にするために定義. violateが起こらないような十分に長いキュー長にする必要あり.)

        self.TASK_C_MAX = (2.0 * 1000) / 1000 # 計算タスクの計算量の最大値を定義
        self.TASK_IMPORTANCE_MAX = 1 # 計算タスクが持つ重要度の最大値を定義

        self.LOCAL_TQ_CALMAX = self.LOCAL_CAPACITY * self.TASK_C_MAX / self.F_VEHICLE # local_tq に並んでいるタスクの計算にかかる時間の最大値
        self.MEC_TQ_CALMAX = self.MEC_CAPACITY * self.TASK_C_MAX / self.F_MEC # mec_tq に並んでいるタスクの計算にかかる時間の最大値

        # 行動空間 (action space) を定義
        n_actions = 2 # アクション数 (localでの処理, MECへのオフロード)
        self.action_space = gym.spaces.Discrete(n_actions) # アクションは離散的なのでDiscrete. 連続的であればBoxで定義.

        # 観測空間 (observation space) を定義
        # 観測は, (localでの処理にかかる処理時間, mecへのオフロードでかかる処理時間, 計算タスクの計算量, 計算タスクの重要度)
        low_lim = np.array([0, 0, 0, 0], dtype=np.float32) # 観測空間の最低値
        high_lim = np.array([self.LOCAL_TQ_CALMAX, self.MEC_TQ_CALMAX, self.TASK_C_MAX, self.TASK_IMPORTANCE_MAX], dtype=np.float32) # 観測空間の最大値
        self.observation_space = gym.spaces.Box(
            low=low_lim, high=high_lim, shape=(4,), dtype=np.float32
        )

        # システム内変数の初期化
        self.time = 1 # システム内の時刻
        self.vehicle_q = deque() # vehicle_q は, システム内の車両のリスト
        for i in range(self.VEHICLE_NUM):
            self.vehicle_q.append(Vehicle(self.time, i+1, 1, 1, 1, 1, self.ARRIVAL_RATE)) # 車両をVEHICLE_NUMだけ生成し, vehicle_qにappendしていく.
        self.vehicle_trajectory = [] # 車両の交通流データを格納するリスト. def loadVehicleTrajectory でこのリストに交通流データを格納していく.
        self.processed_vehicle_num = 0 # 現在のタイムスロットでオフロード戦略の決定が完了した車両数(processed_vehicle_num)を0として初期化
        self.mec_q = deque() # mec_qは, システム内のMECのリスト
        for i in range(self.MEC_NUM):
            self.mec_q.append(Mec(i+1, 1, 300, 500)) # MECをMEC_NUMだけ生成し, mec_qにappendしていく
        self.vehicle = self.vehicle_q[0] # 現在のstepで処理対象になっている車両インスタンスを紐付ける (クリアなコードにするためにこのように実装. 本質的には必要ない.)
        self.mec = self.searchNearestMec(self.vehicle) # オフロード先のMECインスタンスを紐付ける
        self.score_now = 0 # 現在のタイムスロットでのscore
        self.score_prev = 0 # 1つ前のタイムスロットでのscore
             #score = (処理に成功した高重要度タスク数 * IMPORTANCE_RATIO + 処理に成功した低重要度タスク数) / (発生した高重要度タスク数 * IMPORTANCE_RATIO + 発生した低重要度タスク数)
             #score_nowは現在のタイムスロットの評価を表す. どれだけタスクの処理に成功したか, の割合を意味する. -> 長期的報酬の計算に使用
        self.process_time_now = 0 # 現在のタイムスロットで, タスク処理にかかった時間の合計 (現時点では未使用, 長期的報酬の計算などに使用する可能性を考慮し, 実装を残しておく)
        self.process_time_prev = 0 # 1つ前のタイムスロットで, タスク処理にかかった時間の合計 (現時点では未使用, 長期的報酬の計算などに使用する可能性を考慮し, 実装を残しておく)
        self.energy_consumption_now = 0 # 1つ前のタイムスロットでのエネルギー消費の合計 (現時点では未使用, 長期的報酬の計算などに使用する可能性を考慮し, 実装を残しておく)
        self.energy_consumption_prev = 0 # 現在のタイムスロットでのエネルギー消費の合計 (現時点では未使用, 長期的報酬の計算などに使用する可能性を考慮し, 実装を残しておく)
        self.start_flag = False # システム内の時間が終了時刻に達したか, のbool値. 初期化としてFalseに設定.
        #self.taskset = self.createTaskset() # 現在のタイムスロットでのタスクセットを生成

        # 評価のための変数の初期化
        self.violate_counter_h = 0 # 高重要度タスクがタスクキューに入らなかった回数
        self.violate_counter_l = 0 # 低重要度タスクがタスクキューに入らなかった回数
        self.fail_counter_h = 0 # 遅延制約内での処理に失敗した高重要度タスクの数
        self.fail_counter_l = 0 # 遅延制約内での処理に失敗した低重要度タスクの数
        self.success_counter_h = 0 # 遅延制約内での処理に成功した高重要度タスクの数
        self.success_counter_l = 0 # 遅延制約内での処理に成功した低重要度タスクの数
        self.num_h = 0 # 生成された高重要度タスクの数
        self.num_l = 0 # 生成された低重要度タスクの数
        self.total_process_time_h = 0 # 高重要度タスクの処理にかかった時間の合計
        self.total_process_time_l = 0 # 低重要度タスクの処理にかかった時間の合計
        self.total_process_time = 0 # タスクの処理にかかった時間の合計
        self.total_utility = 0 # システム効用の合計
        

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)

        # 終了時刻に達したあとにresetが呼ばれた場合は, カウンタ系を含めた全ての変数を初期化する必要がある.
        if self.start_flag == True:
            self.violate_counter_h = 0
            self.violate_counter_l = 0
            self.fail_counter_h = 0
            self.fail_counter_l = 0
            self.success_counter_h = 0
            self.success_counter_l = 0
            self.num_h = 0
            self.num_l = 0
            self.total_process_time_h = 0
            self.total_process_time_l = 0
            self.total_process_time = 0
            self.time = 1
            self.vehicle_q.clear()
            for i in range(self.VEHICLE_NUM):
                self.vehicle_q.append(Vehicle(self.time, i+1, 1, 1, 1, 1, self.ARRIVAL_RATE))
            self.vehicle_trajectory = []
            self.mec_q.clear()
            for i in range(self.MEC_NUM):
                self.mec_q.append(Mec(i+1, 1, 300, 500))
            self.time = 1
            self.score_now = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.score_prev = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.total_utility = 0
            self.process_time_now = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.process_time_prev = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.energy_consumption_prev = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.energy_consumption_now = 0 # 1タイムスロットごとに随時更新しているので, 終了時刻による初期化でよい
            self.start_flag = False

        # 1タイムスロットごとにresetする変数.
        #self.taskset = self.createTaskset() # 次のタイムスロットのタスクを生成
        #self.sortTaskset()
        #self.task = self.taskset[0] # 次のタイムスロットで初めに処理されるタスクを設定
        self.createTaskForVehicle(self.time) # 各車両にタスクを生成
        self.vehicle = self.vehicle_q[0] # 現在のstepで処理対象になっている車両インスタンスを紐付ける (クリアなコーディングのためにこのように実装. 本質的には不要.)
        self.mec = self.searchNearestMec(self.vehicle) # オフロード先のMECインスタンスを紐付ける (後ほどstep内で最適なMECに更新. 本質的には不要.)
        self.processed_vehicle_num = 0 # 現在のタイムスロットでオフロード戦略の決定が完了した車両数(processed_vehicle_num)を0に
        self.num_h_now = 0 # 現在のタイムスロットで発生した高重要度タスクの数を0に
        self.num_l_now = 0 # 現在のタイムスロットで発生した低重要度タスクの数を0に
        self.local_tq_caltime = self.calProcessTime(self.LOCAL) # 観測で用いるlocalキューに並ぶタスクの処理時間を, 車両v0のlocalタスクキューの値に.
        self.mec_tq_caltime = self.calProcessTime(self.MEC) # 観測で用いるMECキューに並ぶタスクの処理時間を計算.

        # reset() は観測とinfoを返す
        return np.array([self.local_tq_caltime, self.mec_tq_caltime, self.vehicle.task.compute_size, self.vehicle.task.importance]).astype(np.float32), {}  # infoは空白のdictとする


    # タイムスロットを跨いだときに, 各車両に対して計算タスクを生成
    def createTaskForVehicle(self, time):
        for v in self.vehicle_q: # vehicle_q 内の各車両に対してタスクを生成
            v.createTask(self.time, self.ARRIVAL_RATE)

    # オフロード戦略(action)をとったときの, エネルギー消費の値を計算
    def calEnergyComsumption(self, action):
        if (action == self.LOCAL):
            cal_time = self.vehicle.task.compute_size / self.F_VEHICLE 
            return cal_time * (self.F_VEHICLE ** 2)
        else:
            cal_time = self.vehicle.task.compute_size / self.F_MEC
            return cal_time * (self.F_MEC ** 2)


    # タスクの処理時間とエネルギー消費をジョイントした, Utilityを計算する関数
    def calUtility(self, action):
        process_time = self.calProcessTime(action)
        energy_comsumption = self.calEnergyComsumption(action)
        if self.vehicle.task.importance == 1:
            #utility = (self.ALPHA * (self.REWARD + (self.vehicle.task.max_delay - process_time)) * self.IMPORTANCE_RATIO) - (1 - self.ALPHA) * energy_comsumption
            utility = self.ALPHA * self.REWARD * self.IMPORTANCE_RATIO - (1 - self.ALPHA) * energy_comsumption
        else:
            #utility = (self.ALPHA * (self.REWARD + (self.vehicle.task.max_delay - process_time)) * 1) - (1 - self.ALPHA) * energy_comsumption
            utility = self.ALPHA * self.REWARD * 1 - (1 - self.ALPHA) * energy_comsumption
        return utility


    # タスクの処理にかかる時間(=process_time)を計算
    # def appendTask(self, action, vehicle_no) によってタスクキューに計算タスクをappendしてから使うことに注意.
    def calProcessTime(self, action):
        # actionがローカルでの処理の場合 -> タスクの計算にかかる時間のみ
        if action == self.LOCAL:
            # self.vehicleは現在のstepで処理対象となっている車両のNoを表す. 処理対象となっている車両のローカルタスクキューを計算に用いる.
            calculate_time = self.calTotalsize(self.vehicle.local_tq) / self.F_VEHICLE
            process_time = calculate_time
        # actionがMECへのオフロードである場合 -> タスクの計算にかかる時間 + タスクの入力データの送信にかかる時間
        else:
            calculate_time = self.calTotalsize(self.mec.mec_tq) / self.F_MEC
            communicate_time = self.calCommTime(self.vehicle, self.mec)
            process_time = calculate_time + communicate_time
        return process_time


    # def calProcessTime~~ 内で用いる関数. タスクキューに並ぶ計算タスクの総計算量を計算.
    def calTotalsize(self, tq): # tq (task queue) は local_tq か mec_tq のいずれか
        total_size = 0
        for t in tq:
            total_size = total_size + t.compute_size
        return total_size
    
    def isCommunicatable(self, vehicle, mec): # 通信対象のMECと車両を引数として取り, 通信可能であればTrue, 不可能ならFalseを返す関数
        l = math.sqrt((vehicle.x - mec.x) ** 2 + (vehicle.y - mec.y) ** 2) # 通信距離を算出
        if l <= mec.range: # MECの通信可能範囲に車両が存在している場合
            return True
        else: # MECの通信可能範囲に車両が存在していない場合
            return False
    
    def searchNearestMec(self, vehicle): # 車両に対して最も近いMECを探し, そのMECインスタンスを返す関数
        min_l = 100000000 # 十分に大きい値を設定
        nearest_mec = self.mec_q[0] # ID 1 のMECを初期値として設定
        for mec in self.mec_q:
            l = math.sqrt((vehicle.x - mec.x) ** 2 + (vehicle.y - mec.y) ** 2) # 通信距離を算出
            if l < min_l: # 距離lの最小値を下回る距離であれば更新
                min_l = l
                nearest_mec = mec
        return nearest_mec # 最小距離にあるMECをreturn
    

    def calCommTime(self, vehicle, mec): # 通信対象のMECと車両を引数として取り, 通信時間を返す関数
        input_size = vehicle.task.input_size # 車両が持つ計算タスクの入力データ量
        rate = self.calCommRate(vehicle, mec)
        rate = 6 # 一旦, 決め打ちで6Mbpsの伝送レートと仮定 (後ほど要修正)
        return input_size / rate # 計算タスクの入力データ量を通信レートで割り, 通信時間を算出
    
    def calCommRate(self, vehicle, mec): # 通信対象のMECと車両を引数として取り, 伝送レートを返す関数
        loss = self.calCommLoss(vehicle, mec) # 通信対象のMECと車両の伝送ロスを計算 (自由空間伝搬損失を仮定)
        gain = self.A_GAIN_v * self.A_GAIN_r / loss # 車両側アンテナの絶対利得 * RSU側アンテナの絶対利得 / 伝送ロス -> チャネルゲインを算出
        rate = self.BANDWIDTH * (1 / (1 + self.calLinkNum(self.mec))) * math.log2(1 + (self.T_POWER * gain) / (self.calNoise() + self.T_POWER * gain)) # シャノンの定理より, 伝送レートを算出
        return rate

    def calCommLoss(self, vehicle, mec): # 通信対象のMECと車両を引数として取り, 伝送ロスを計算 (自由空間伝搬損失を仮定)
        l = math.sqrt((vehicle.x - mec.x) ** 2 + (vehicle.y - mec.y) ** 2) # 通信距離を算出 (m)
        wavelength = (3.0 * (10 ** 8)) / (self.WAVEGHz * (10 ** 9)) # 波長を算出 (m)
        loss = (4 * math.pi * l / wavelength) ** 2
        return loss
    
    def calNoise(self): # 外部からのノイズ(dBm)をW単位に変換
        noise = self.dbmToWatt(self.NOISE)
        return noise

    def dbmToWatt(self, dbm): # dBmをWに変換する関数
        return 10 ** ((dbm - 30) / 10)
    
    def calLinkNum(self, mec): # 引数として取ったMECと通信している車両数を計算
        return mec.link

    # タスクキューに計算タスクを追加する関数. violateが起こったかどうかを返り値とする.
    def appendTask(self, action):
        violation = False

        if action == self.LOCAL:
            # 以下のif文が成立する場合, local_tqは最大長になってしまっていて, 新たにタスクをappendできない -> violate判定
            if len(self.vehicle.local_tq) >= self.LOCAL_CAPACITY:
                violation = True
                # violateが発生した場合にはfail_counterを増やす (タスクの処理としては失敗と判定)
                if self.vehicle.task.importance == 1:
                    self.fail_counter_h = self.fail_counter_h + 1
                    self.violate_counter_h = self.violate_counter_h + 1
                    self.num_h = self.num_h + 1
                    self.num_h_now = self.num_h_now + 1
                else:
                    self.fail_counter_l = self.fail_counter_l + 1
                    self.violate_counter_l = self.violate_counter_l + 1
                    self.num_l = self.num_l + 1
                    self.num_l_now = self.num_l_now + 1
            # violateが起きていない (=タスクキューが最大長ではない) 場合は, local_tqにタスクをappendする.
            else:
                self.vehicle.local_tq.append(self.vehicle.task)

        elif action == self.MEC:
            # 以下のif文が成立する場合, mec_tqは最大長になってしまっていて, 新たにタスクをappendできない -> violate判定
            if len(self.mec.mec_tq) >= self.MEC_CAPACITY:
                violation = True
                # violateが発生した場合にはfail_counterを増やす (タスクの処理としては失敗と判定)
                if self.vehicle.task.importance == 1:
                    self.fail_counter_h = self.fail_counter_h + 1
                    self.num_h = self.num_h + 1
                    self.num_h_now = self.num_h_now + 1
                else:
                    self.fail_counter_l = self.fail_counter_l + 1
                    self.num_l = self.num_l + 1
                    self.num_l_now = self.num_l_now + 1
            # violateが起きていない (=タスクキューが最大長ではない) 場合は, mec.mec_tqにタスクをappendする.
            else:
                self.mec.mec_tq.append(self.vehicle.task)
        #violateが起こったかどうかを返り値とする. -> 報酬を決定するときに使用
        return violation


    # タスクキュー内の計算タスクを処理する関数. タイムスロットの終わりに呼び出して実行. 1秒を1タイムスロットとしているので, 1秒分のタスク処理を進める.
    def processTask(self):
        # local_tq 内の計算タスクを処理
        for i in range(self.VEHICLE_NUM): # 全車両についてlocal_tqを1秒ぶん処理し, 更新
            time = 0 # time(=処理時間の合計)が1になるまで処理
            temp_time = 0 # 処理の終了判定に使用
            stop = False # 時間が1タイムスロット(=1秒)を越えるかどうかを意味するbool値
            while (stop == False) and (len(self.vehicle_q[i].local_tq) != 0):
                now_processed = self.vehicle_q[i].local_tq[0] # 現在処理対象としている計算タスクを指す. タスクキュー左端に最も時系列的に古い計算タスクが存在しているため, 左端から処理していく.
                temp_time = time + (now_processed.compute_size / self.F_VEHICLE) # 現在対象の計算タスクを終わりまで処理したとして, timeがどうなるか, という仮の値を計算
                if temp_time <= 1: # 現在対象の計算タスクを完全に処理してもまだ処理時間に余裕があるなら, 現在対象の計算タスクは処理完了とする.
                    time = temp_time # 処理時間の合計を更新
                    # 注意 : タスクキュー内, 左端に最も時系列的に古い計算タスクが存在しているため, 左端からpopしていく必要がある.
                    self.vehicle_q[i].local_tq.popleft()
                else: # 現在対象の計算タスクを完全に処理するだけの残り時間がなければ, できるところまで処理して次のタイムスロットへ移行することになる.
                    now_processed.compute_size = now_processed.compute_size - (1 - time) * self.F_VEHICLE # compute_sizeを更新. 残り時間でできる分だけ処理されるため減算.
                    stop = True # 処理の終了判定

        # mec_tq 内の計算タスクを処理
        for i in range(self.MEC_NUM): # 全MECについてmec_tqを1秒ぶん処理し, 更新
            time = 0 # time(=処理時間の合計)が1になるまで処理
            temp_time = 0 # 処理の終了判定に使用
            stop = False # 時間が1タイムスロット(=1秒)を越えるかどうかを意味するbool値
            while (stop == False) and (len(self.mec_q[i].mec_tq) != 0):
                now_processed = self.mec_q[i].mec_tq[0] # 現在処理対象としている計算タスクを指す. タスクキュー左端に最も時系列的に古い計算タスクが存在しているため, 左端から処理していく.
                temp_time = time + (now_processed.compute_size / self.F_MEC) # 現在対象の計算タスクを終わりまで処理したとして, timeがどうなるか, という仮の値を計算
                if temp_time <= 1: # 現在対象の計算タスクを完全に処理してもまだ処理時間に余裕があるなら, 現在対象の計算タスクは処理完了とする.
                    time = temp_time # 処理時間の合計を更新
                    # 注意 : タスクキュー内, 左端に最も時系列的に古い計算タスクが存在しているため, 左端からpopしていく必要がある.
                    self.mec_q[i].mec_tq.popleft()
                else: # 現在対象の計算タスクを完全に処理するだけの残り時間がなければ, できるところまで処理して次のタイムスロットへ移行することになる.
                    now_processed.compute_size = now_processed.compute_size - (1 - time) * self.F_MEC # compute_sizeを更新. 残り時間でできる分だけ処理されるため減算.
                    stop = True # 処理の終了判定


    # 各種, ~~_prevの値を更新する関数
    def updatePrev(self):
        self.process_time_prev = self.process_time_now
        self.energy_consumption_prev = self.energy_consumption_now
        self.score_prev = self.score_now
        return


    def step(self, action):
        
        # actionが不正な値だった場合にはerror. 通常起こらないはず.
        if (action != self.LOCAL) and (action != self.MEC):
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        # ここからが通常の処理
        else:
            violation = self.appendTask(action)
            if violation != True:
                reward = 0
            else:
                reward = -1 # violateが起こった場合には報酬としてペナルティ

        # violateが起こっていなければタスクの処理時間(process_time)を計算し, タスクの遅延制約内に処理できているか判定 -> 短期的報酬の設定
        if violation != True:
            # タスクの処理にかかる時間(=process_time)を計算
            process_time = self.calProcessTime(action)

            if action == self.MEC: # オフロード戦略がMECへのオフロードであれば, 対象MECのリンク数を1増やす
                self.mec.link = self.mec.link + 1

            # 評価のため, タスクの処理時間をこれまでの処理時間の合計に加える
            self.total_process_time = self.total_process_time + process_time

            # process_time_now の算出のため, タスクの処理時間をこれまでのprocess_time_nowに加える
            self.process_time_now = self.process_time_now + process_time

            # タスク処理によるエネルギー消費を算出
            energy_comsumption = self.calEnergyComsumption(action)

            # energy_comsumption_now の算出のため, エネルギー消費をこれまでのenergy_comsumption_nowに加える
            self.energy_consumption_now = self.energy_consumption_now + energy_comsumption

            # タスクの遅延制約内にタスクの処理が完了する場合 (=タスクの処理成功の場合)
            if process_time <= self.vehicle.task.max_delay:
                utility = self.calUtility(action) # 処理成功によるutilityを計算
                self.total_utility = self.total_utility + utility # 評価のため, 通算のUtilityを更新
                reward = utility # 短期的報酬を設定

                # 高重要度タスクが成功した場合
                if self.vehicle.task.importance == 1:
                    # 長期的報酬の計算のため, 現在のタイムスロットのscoreを更新
                    self.score_now = self.score_now + self.REWARD * self.IMPORTANCE_RATIO
                    # 評価のための各種カウンタを更新
                    self.success_counter_h = self.success_counter_h + 1
                    self.num_h = self.num_h + 1
                    self.num_h_now = self.num_h_now + 1
                    # 評価のためのtotal_process_time_hを更新
                    self.total_process_time_h = self.total_process_time_h + process_time

                # 低重要度タスクが成功した場合
                else:
                    # 長期的報酬の計算のため, 現在のタイムスロットのscoreを更新
                    self.score_now = self.score_now + self.REWARD
                    # 評価のための各種カウンタを更新
                    self.success_counter_l = self.success_counter_l + 1
                    self.num_l = self.num_l + 1
                    self.num_l_now = self.num_l_now + 1
                    # 評価のためのtotal_process_time_lを更新
                    self.total_process_time_l = self.total_process_time_l + process_time

            # タスクの遅延制約内にタスクの処理が完了しない場合 (=タスクの処理失敗の場合)
            else:
                # 高重要度タスクが失敗した場合
                if self.vehicle.task.importance == 1:
                    self.fail_counter_h = self.fail_counter_h + 1
                    self.num_h = self.num_h + 1
                    self.num_h_now = self.num_h_now + 1
                    reward = - self.REWARD * self.IMPORTANCE_RATIO # 短期的報酬として, 固定報酬の負の値をrewardとする
                    # 評価のためのtotal_process_time_hを更新
                    self.total_process_time_h = self.total_process_time_h + process_time

                # 低重要度タスクが成功した場合
                else:
                    self.fail_counter_l = self.fail_counter_l + 1
                    self.num_l = self.num_l + 1
                    self.num_l_now = self.num_l_now + 1
                    reward = - self.REWARD # 短期的報酬として, 固定報酬の負の値をrewardとする
                    # 評価のためのtotal_process_time_lを更新
                    self.total_process_time_l = self.total_process_time_l + process_time

        self.processed_vehicle_num = self.processed_vehicle_num + 1 # オフロード戦略の決定が完了した車両数を1増やす
        if self.processed_vehicle_num >= self.VEHICLE_NUM: # オフロード戦略の決定が完了した車両数がシステム内の車両数に達した場合, そのタイムスロット(=episode)は終了
            terminated = True
        else: # 現在のタイムスロットが終了していない場合
            terminated = False
            self.vehicle = self.vehicle_q[self.processed_vehicle_num] # 次の処理対象の車両をself.vehicleに設定
            self.mec = self.searchNearestMec(self.vehicle) # 次の処理対象の車両に対し, 最も近いMECをself.mecに設定
        
        # 次の処理対象になる車両に計算タスクが存在していない場合と, 次の処理対象になる車両と通信可能なMECがない場合には特別な対応が必要
        while ((self.vehicle.exist_task == False) or (self.isCommunicatable(self.vehicle, self.mec) == False)):
            if self.vehicle.exist_task == False: # 次の処理対象になる車両に計算タスクが存在していない場合, その車両の処理はskipする.
                self.processed_vehicle_num = self.processed_vehicle_num + 1 # オフロード戦略の決定が完了した車両数を1増やす (車両に計算タスクが存在しなかったため)
            else: # 次の処理対象になる車両と通信可能なMECがない場合, タスクをローカルで処理
                self.appendTask(self.LOCAL) # オフロード戦略は強制的にLOCALとなる.
                self.processed_vehicle_num = self.processed_vehicle_num + 1 # オフロード戦略の決定が完了した車両数を1増やす

            if self.processed_vehicle_num >= self.VEHICLE_NUM: # オフロード戦略の決定が完了した車両数がシステム内の車両数に達した場合, そのタイムスロット(=episode)は終了
                terminated = True
                break
            else:
                terminated = False # 全車両のオフロード戦略が未決定であれば, まだタイムスロット内での処理であるから, terminated は False
                # processed_vehicle_numには, オフロード戦略の決定が完了した車両数が入っているため, vehicle_qのindexとして使用することで, 次の処理対象の車両を示すことができる.
                # 注意：リストのindexは0オリジンである.
                self.vehicle = self.vehicle_q[self.processed_vehicle_num] # 処理対象の車両を改めて設定
                self.mec = self.searchNearestMec(self.vehicle) # 処理対象の車両に最も近いMECを改めて設定

        #terminated = bool(self.processed_vehicle_num >= self.VEHICLE_NUM) # オフロード戦略の決定が完了した車両数がシステム内の車両数に達した場合, そのタイムスロット(=episode)は終了
        truncated = False  # truncated は今回は定めない

        # タイムスロットの終わりであれば, 以下のコードを実行
        if terminated:
            # score_now について, ここまでタスクが成功するごとに, REWARD (* IMPORTANCE_RATIO) を加算しているため, 全ての処理に成功した場合(=scoreの最大値)で除算し, scoreを算出する.
            self.score_now = self.score_now / (self.num_h_now * self.REWARD * self.IMPORTANCE_RATIO + self.num_l_now * self.REWARD)
            # ゼロ除算を避けるため, utilityが0の場合は非常に小さい値にすり替える
            if self.score_now == 0:
                self.score_now = 0.001
                print("WARN : score_now == 0 is True at time ", self.time)
            # システム内の時間が1でなければ, 前のタイムスロットが存在 -> 長期的報酬を計算
            if self.time != 1:
                # 長期的報酬をrewardに加算 <- Greedyアルゴリズムの評価のため, ここでは不要.
                # reward = reward + ((self.score_now - self.score_prev) / self.score_prev) * self.BETA
                # rewardが非常に大きい値になってしまい, 局所解に陥ることを防ぐため, reward clippingを導入
                if reward > 50.0:
                    reward = 50.0
                elif reward < -50.0:
                    reward = -50.0
            # 各種prevを更新
            self.updatePrev()
            self.processTask()
            self.time = self.time + 1 # システム内の時間を進める

            # 注意：次のタイムスロット(=episode)の初期観測は, reset()によって与えられるため, ここでは観測の情報を更新しない.

            if self.time > self.TERMINATE_TIME: # 終了時刻に達していれば, start_flagをTrueにしてからresetが呼ばれるようにする.
                self.start_flag = True
            else:
              self.start_flag = False # 念の為, 明示的にコーディング

        # タイムスロットの終わりではないなら, 次の車両の処理へ移行
        else:
            # self.vehicleを更新してからcalProcessTimeを呼ぶことで, 次のstepで処理対象となる車両のlocal_tqの情報を, 観測として与える
            self.local_tq_caltime = self.calProcessTime(self.LOCAL)
            self.mec_tq_caltime = self.calProcessTime(self.MEC)

        # infoは今回は使用しないため, 空のdictとする.
        info = {}

        return (
            np.array([self.local_tq_caltime, self.mec_tq_caltime, self.vehicle.task.compute_size, self.vehicle.task.importance]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass


random.seed()
random.seed(100)
env = SimpleOffloadEnvForGreedy()
#env = SimpleOffloadEnv()
obs, info = env.reset()
n_steps = TERMINATE_TIME * VEHICLE_NUM
total_reward = 0
time = 1

for step in range(n_steps):
    action = greedyAction(env, 1)
    #print("Time : ", env.time, " / Task importance : ", env.task.importance, " / action : ", action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward = total_reward + reward

    if done:
        obs, info = env.reset()
        if env.time >= TERMINATE_TIME:
            print("-- Evaluation for greedy algorithm 2 --")
            print("violate_counter: ", env.violate_counter)
            print("success_counter_h: ", env.success_counterh)
            print("success_counter_l: ", env.success_counter_l)
            print("fail_counter_h: ", env.fail_counter_h)
            print("fail_counter_l: ", env.fail_counter_l)
            print("ave_reward =", round((total_reward / n_steps), 5))
            #print("average process time (high importance) : ", round((env.unwrapped.total_process_time_h/(env.unwrapped.success_counter_h + env.unwrapped.fail_counter_h)), 1))
            #print("average process time (low importance) : ", round((env.total_process_time_l/(env.success_counter_l + env.fail_counter_l)), 1))
            print("success rate (high importance) :", round((env.success_counter_h/env.num_h) * 100, 1), "%")
            print("success rate (low importance) :", round((env.success_counter_l/env.num_l) * 100, 1), "%")
            print("score : ", env.success_counter_h*env.IMPORTANCE_RATIO + env.success_counter_l)
            print("")
            break
        obs, info = env.reset()
"""