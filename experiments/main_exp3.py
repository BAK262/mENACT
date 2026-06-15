# Author: Ming Li <liming16@tsinghua.org.cn>
# Date: 2023/12/25
# Requirements: python=3.8, psychopy(2023.02), moviepy

from utils_exp import Experiment3
import serial
import sys

# Initialize the program (press [Ctr]+[Alt]+[Q] to quit)
exp = Experiment3(exp_dir=sys.path[0],
                  stimuli_screen=1,  # the TV
                  observe_screen=0,  # the monitor
                  port=serial.Serial('COM3', 9600),
                  microphone='MEETEASY USB AUDIO',
                  camera='Logitech StreamCam',
                  # (1920,1080),(1280,720),(960,540),(848,480),(640,360),(320,240)
                  camFrameSize=(1920, 1080),
                  camFrameRate=30)  # 60 fps，30 fps，24 fps，20 fps, 15 fps，10 fps，7.5 fps，5 fps

# Welcome page
# Output in [data_dir\\subject_id]:
#       subjectFile - i.e., [subject_info.csv]
exp.start()
exp.send_trigger([48, 48, 48, 48, 48])
exp.show_instruction(wait='space', text=exp.welcome_guide)

# Practice session
exp.show_instruction(wait='space', text=exp.practice_guide)
exp.show_instruction(wait=1, text='练习环节')
exp.show_instruction(wait='space', text=exp.rehersal_guide('neutral'))
exp.show_script(emotion='neutral', log=False)
exp.show_instruction(wait='space', text=exp.act_guide)
exp.act_task(duration=180, prepare=3, log=False, save=False)
exp.get_ratings(log=False)
exp.show_instruction(wait='space', text=exp.math_guide)
exp.show_fixation(wait=3)
exp.do_math(n=5, log=False)
exp.show_instruction(wait='space', text='练习环节完毕，请呼叫主试。')

# Formal session
# Output in [data_dir\\subject_id]:
#       ratingFile - e.g., [memory_20231227220809_rating.csv]
#       mathFile - e.g., [memory_202312270809_math.csv]
for block in exp.blocks:
    exp.block_idx += 1
    for trial in block:
        print(f'\n------ Target Emotion: {trial} ------')
        exp.trial_idx += 1
        exp.trial_log.at[exp.trial_idx, 'targetEmotion'] = trial
        exp.show_instruction(wait='space', text=exp.rehersal_guide(trial))
        print('Rehersal start.')
        exp.show_script(emotion=trial, log=True)
        print('Rehersal end.')
        exp.show_instruction(wait='space', text=exp.act_guide)
        print('Acting start.')
        exp.act_task(duration=180, prepare=3, emotion=trial)
        print('Acting end.')
        print('Rating start.')
        exp.get_ratings()
        print('Rating end.')
    if exp.block_idx < 3:
        exp.show_instruction(wait='space', text=exp.math_guide)
        exp.show_fixation(wait=3)
        exp.do_math()
exp.send_trigger([49, 49, 49, 49, 49])
exp.show_instruction(wait='space', text='实验结束，请呼叫主试。')
exp.show_instruction(wait='escape', text='（1）按【ESC】退出画面\n（2）停止记录并保存数据\n（3）为被试摘掉头帽\n（4）等待命令行出现“(psychopy) <your-prompt>”\n（5）最后，再关闭 VS Code！')
exp.end()
