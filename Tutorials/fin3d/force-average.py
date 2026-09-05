import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_and_plot():
    file_name = 'IB_Fin_1.csv'

    # 检查文件是否存在
    if not os.path.exists(file_name):
        print(f"错误: 找不到文件 '{file_name}'。请确保代码与CSV文件在同一目录下。")
        return

    # 读取CSV文件
    print(f"正在读取 {file_name} ...\n")
    df = pd.read_csv(file_name)

    # 获取总的时间区间
    total_t_min = df['time'].min()
    total_t_max = df['time'].max()

    print("="*40)
    print(f"数据加载成功！")
    print(f"总的时间区间为: {total_t_min:.6f} 秒 到 {total_t_max:.6f} 秒")
    print("="*40)

    # 设置绘图字体，防止中文乱码 (如果你系统不支持以下字体，图表将自动回退到默认英文字体)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    while True:
        try:
            print("\n请输入你想计算的时间段 (输入 'q' 退出):")
            start_input = input("起始时间 (start_time): ")
            if start_input.lower() == 'q':
                break

            end_input = input("结束时间 (end_time): ")
            if end_input.lower() == 'q':
                break

            t_start = float(start_input)
            t_end = float(end_input)

            if t_start >= t_end:
                print("错误: 结束时间必须大于起始时间，请重新输入。")
                continue

            # 筛选该时间段内的数据
            mask = (df['time'] >= t_start) & (df['time'] <= t_end)
            filtered_df = df[mask]

            if len(filtered_df) < 2:
                print("错误: 该时间段内的数据点太少（少于2个），无法计算或绘图。请扩大时间范围。")
                continue

            # 提取时间和力的数据
            t_array = filtered_df['time'].values
            fx_array = filtered_df['Fx'].values

            # 计算实际截取到的时间跨度
            actual_time_span = t_array[-1] - t_array[0]

            # 1. 积分时间平均值 (最准确的物理定义)
            if hasattr(np, 'trapezoid'):
                integral = np.trapezoid(fx_array, t_array)
            else:
                integral = np.trapz(fx_array, t_array)

            time_avg_fx = integral / actual_time_span

            # 2. 算数平均值 (仅供参考)
            arithmetic_avg_fx = np.mean(fx_array)

            print("-" * 40)
            print(f"区间 [{t_array[0]:.6f}, {t_array[-1]:.6f}] 的计算结果:")
            print(f"截取到的数据点数量: {len(filtered_df)}")
            print(f"▶ 精确时间平均值 (积分法): {time_avg_fx:.6f}")
            print(f"  (参考) 简单算数平均值: {arithmetic_avg_fx:.6f}")
            print("-" * 40)

            # 开始绘图
            plt.figure(figsize=(10, 5)) # 设置图片大小

            # 画出 Fx 随时间变化的真实曲线
            plt.plot(t_array, fx_array, label='Fx 真实曲线', color='royalblue', linewidth=1.5)

            # 画出表示时间平均值的红虚线
            plt.axhline(y=time_avg_fx, color='red', linestyle='--', linewidth=2,
                        label=f'时间平均值 = {time_avg_fx:.4f}')

            # 填充曲线和平均值之间的区域（选作视觉增强，让波动更直观）
            plt.fill_between(t_array, fx_array, time_avg_fx, color='royalblue', alpha=0.1)

            # 设置图表标签和标题
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Fx', fontsize=12)
            plt.title(f'Fx 时间历程曲线 ({t_array[0]:.3f}s ~ {t_array[-1]:.3f}s)', fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(loc='best', fontsize=11)

            print(">> 提示: 请关闭弹出的图像窗口，即可继续下一次计算。")
            # 显示图像 (程序会在此处暂停，直到你关闭图像窗口)
            plt.show()

        except ValueError:
            print("错误: 输入无效，请输入数字。")

if __name__ == "__main__":
    calculate_and_plot()
