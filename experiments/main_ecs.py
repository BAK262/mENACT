
# Author: Ming Li <liming16@tsinghua.org.cn>
# Date: 2023/12/25
# Requirements: python=3.8, psychopy(2023.02), moviepy
#
# Individual-level emotion concept similarity (ECS) task — PsychoPy only, no fNIRS.
# Output: trait_ecs.csv in data/all_raw/<subject_id>/

from utils_exp import EmotionConceptSimilarityTask
import sys

# Initialize the program (press [Ctr]+[Alt]+[Q] to quit)
exp = EmotionConceptSimilarityTask(exp_dir=sys.path[0],
                  stimuli_screen=1,  # the TV
                  observe_screen=0)  # the monitor

# Welcome page
# Output in [data_dir\\subject_id]:
#       subjectFile - i.e., [subject_info.csv]
exp.start()
exp.show_instruction(wait='space', text=exp.welcome_guide)

# Formal session
# Output in [data_dir\\subject_id]:
#       trait_ecs.csv (pairwise similarity ratings)
for _ in range(exp.n_pairs):
    exp.judgment()
    print(f'--- {exp.trial_idx+1} / {exp.n_pairs} 已完成 ---')
exp.show_instruction(wait='space', text='实验结束，请呼叫主试。')
exp.end()
