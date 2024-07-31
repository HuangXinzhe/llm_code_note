# bunch of less exciting, common utilities we'll use in multiple files
import time
from math import log, cos, sin, pi

# -----------------------------------------------------------------------------
# random number generation

def box_muller_transform(u1, u2):
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    # This is using the Basic form of the Box-Muller transform
    # u1 and u2 are simple floats in [0, 1)
    # z1 and z2 are standard normal random variables
    # 实现了基本形式的 Box-Muller 变换，用于将两个 [0, 1) 区间上的均匀随机数 u1 和 u2，
    # 转换为标准正态分布的随机数 z1 和 z2。
    z1 = (-2 * log(u1)) ** 0.5 * cos(2 * pi * u2)
    z2 = (-2 * log(u1)) ** 0.5 * sin(2 * pi * u2)
    return z1, z2

# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc.
# RNG 类提供了一个完全确定性的随机数生成器，实现了均匀分布或正态分布随机数的生成。通过 Box-Muller 变换和xorshift*算法实现生成正态分布和均匀分布的随机数，确保了生成器的可重复性和可控性。
class RNG:
    def __init__(self, seed):
        # 接受一个种子 seed，用于初始化随机数生成器的状态 self.state，使得生成器是确定性的。
        self.state = seed

    def random_u32(self):
        # 实现了 xorshift* 随机数生成算法。
        ## 使用按位操作（^ 和 >>, <<）更新 self.state。
        ## 返回一个 32 位无符号整数，确保输出在 [0, 2^32-1] 范围内。
        # 使用 & 0xFFFFFFFFFFFFFFFF 确保结果在 64 位范围内（类似于在 C 中将结果强制转换为 uint64）。
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF 
        # 上一步的操作：
        ## 1.右移 self.state 12 位，并与 0xFFFFFFFFFFFFFFFF 按位与，确保结果在 64 位范围内。
        ## 2.使用按位异或操作 ^= 更新 self.state。
        ## 这一操作混合了 self.state 的高位和低位，增加了随机性。 
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        # 使用 & 0xFFFFFFFF 确保结果在 32 位范围内（类似于在 C 中将结果强制转换为 uint32）。
        ## 将 self.state 乘以一个常数 0x2545F4914F6CDD1D。
        ## 这个常数是经过选择的，可以确保生成的随机数具有良好的分布特性。
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # 将 random_u32 生成的 32 位无符号整数右移 8 位，并除以 2^24（16777216.0）
        # 得到 [0, 1) 区间的浮点数。
        return (self.random_u32() >> 8) / 16777216.0

    def rand(self, n, a=0, b=1):
        # 生成 n 个 [a, b) 区间的均匀随机数，并返回一个列表。
        ## 使用列表推导式调用 random 函数生成 n 个随机数，并将它们线性映射到 [a, b) 区间。
        return [self.random() * (b - a) + a for _ in range(n)]

    def randn(self, n, mu=0, sigma=1):
        # 生成 n 个服从正态分布 N(mu, sigma^2) 的随机数，并返回一个列表。
        out = []
        for _ in range((n + 1) // 2):
            u1, u2 = self.random(), self.random()
            z1, z2 = box_muller_transform(u1, u2) # # 使用 Box-Muller 变换生成两个标准正态分布的随机数 z1 和 z2，并将它们扩展到输出列表 out。
            out.extend([z1 * sigma + mu, z2 * sigma + mu]) # 乘以 sigma 并加上 mu 以调整到期望的均值和标准差。
        out = out[:n] # 如果 n 是奇数，截断列表 out 以确保返回 n 个随机数。
        return out

# -----------------------------------------------------------------------------
# StepTimer for timing code

class StepTimer:
    def __init__(self, ema_alpha=0.9):
        self.ema_alpha = ema_alpha
        self.ema_time = 0
        self.corrected_ema_time = 0.0
        self.start_time = None
        self.step = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        iteration_time = end_time - self.start_time
        self.ema_time = self.ema_alpha * self.ema_time + (1 - self.ema_alpha) * iteration_time
        self.step += 1
        self.corrected_ema_time = self.ema_time / (1 - self.ema_alpha ** self.step) # bias correction

    def get_dt(self):
        return self.corrected_ema_time