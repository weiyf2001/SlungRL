import sys
import types

# =====================================================================
# 【黑科技：运行时模块欺骗】
# 在一切开始之前，我们在内存中伪造一个只剩空壳的 mujoco_py。
# 这样可以阻止 gymnasium 触发真实的、会报错的 mujoco-py 初始化代码。
# =====================================================================
fake_mujoco_py = types.ModuleType("mujoco_py")


# 伪造一个异常类，防止 gymnasium 内部有捕获异常的逻辑导致找不到属性
class DummyException(Exception):
    pass


fake_builder = types.ModuleType("builder")
fake_builder.MujocoException = DummyException
fake_mujoco_py.builder = fake_builder
fake_mujoco_py.MujocoException = DummyException

# 强行将其注册到系统的已加载模块字典中
sys.modules["mujoco_py"] = fake_mujoco_py
sys.modules["mujoco_py.builder"] = fake_builder

# =====================================================================
# 欺骗完成，现在可以安全导入 gymnasium 并使用官方的 mujoco 3.2.7 了！
# =====================================================================
import gymnasium as gym
import time


def test_fps(env_id, num_envs, steps_per_env=1000):
    # 创建异步向量环境
    envs = gym.vector.AsyncVectorEnv([
        lambda: gym.make(env_id) for _ in range(num_envs)
    ])

    envs.reset()
    start_time = time.time()

    for _ in range(steps_per_env):
        actions = envs.action_space.sample()
        envs.step(actions)

    end_time = time.time()
    envs.close()

    elapsed_time = end_time - start_time
    fps = (num_envs * steps_per_env) / elapsed_time
    return fps


if __name__ == "__main__":
    ENV_ID = "Ant-v4"

    # 测试的并行数量
    num_envs_list = [1, 2, 4, 8, 16,32,64,128]
    STEPS = 1000

    print(f"正在测试环境: {ENV_ID} (已成功绕过旧版 mujoco-py 依赖)")
    print(f"每个环境运行步数: {STEPS}")
    print("-" * 35)
    print(f"{'并行环境数':<12} | {'FPS':<15}")
    print("-" * 35)

    for num_envs in num_envs_list:
        try:
            fps = test_fps(ENV_ID, num_envs, STEPS)
            print(f"{num_envs:<15} | {fps:.2f}")
        except Exception as e:
            print(f"{num_envs:<15} | 运行失败: {e}")