# Author: Ming Li <liming16@tsinghua.org.cn>
# Date: 2023/12/25
# Requirements: python=3.8, psychopy(2023.02), moviepy

from utils_exp import Experiment1
import serial
import sys

# Initialize the program (press [Ctr]+[Alt]+[Q] to quit)
exp = Experiment1(exp_dir=sys.path[0],
                  stimuli_screen=1,
                  observe_screen=0,
                  port=serial.Serial('COM3', 9600))

# Welcome page
# Output in [data_dir\\subject_id]:
#       subjectFile - i.e., [subject_info.csv]
exp.start()
exp.send_trigger([48, 48, 48, 48, 48])
exp.show_instruction(wait='space', text=exp.welcome_guide)

# Practice session
exp.show_instruction(wait='space', text=exp.practice_guide)
exp.show_instruction(wait=1, text='练习环节')
exp.show_fixation(wait=3)
exp.play_video(video_idx=13, log=False)
exp.get_ratings(log=False)
exp.show_instruction(wait='space', text=exp.math_guide)
exp.show_fixation(wait=3)
exp.do_math(n=5, log=False)
exp.show_instruction(wait='space', text='练习环节完毕，请呼叫主试。')

# Formal session
# Output in [data_dir\\subject_id]:
#       ratingFile - e.g., [video_20231227220809_rating.csv]
#       mathFile - e.g., [video_202312270809_math.csv]
for block in exp.blocks:
    exp.block_idx += 1
    for trial in block:
        exp.show_instruction(wait='space', text='按【空格/SPACE】键观看下一段视频')
        exp.show_fixation(wait=3)
        exp.play_video(video_idx=trial)
        exp.get_ratings()
    if exp.block_idx < 7:
        exp.show_instruction(wait='space', text=exp.math_guide)
        exp.show_fixation(wait=3)
        exp.do_math()
exp.send_trigger([49, 49, 49, 49, 49])
exp.show_instruction(wait='space', text='视频播放完毕，请呼叫主试。')
exp.show_instruction(wait='escape', text='（1）按【ESC】退出画面\n（2）停止记录并保存数据\n（3）为被试摘掉头帽\n（4）关闭 VS Code')
exp.end()
