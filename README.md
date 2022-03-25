# KG-A2C
This repository contains a reference implementation KG-A2C as mentioned in [Graph Constrained Reinforcement Learning for Natural Language Action Spaces](https://openreview.net/forum?id=B1x6w0EtwH), that has been modified for use with the [ScienceWorld](https://www.github.com/allenai/ScienceWorld) environment.

# Quickstart
Clone the repository:
```bash
git clone git@github.com:cognitiveailab/kga2c-scienceworld.git
cd kga2c-scienceworld
```

Install Dependencies:
```bash
conda create --name kga2c-scienceworld python=3.7
conda activate kga2c-scienceworld
pip install -r requirements.txt
```
You may want to install the pytorch manually if your GPU does not support CUDA 11.

Train KG-A2C
```bash
cd kga2c
mkdir logs
python train.py --task_num=0 --batch_size=8 --simplification_str=easy --stuck_steps=100 --reset_steps=100 --steps=100000 --test_interval=1000 --seed=0 --output_dir logs
```

Here:
- **task_num:** The ScienceWorld task index (0-29). *See **task list** below*
- **batch_size:** The number of environment threads to simultaneously use during training (8 is a common number)
- **simplification_str:** The ScienceWorld simplification string
- **stuck_steps:** If the agent continuously generates stuck_steps number of invalid actions, the environment will reset
- **reset_steps:** the maximum steps per episode
- **steps:** the maximum number of steps
- **test_interval:** the number of steps between evaluations
- **seed:** random seed
- **output_dir:** output directory

## ScienceWorld Task List
```
TASK LIST:
    0: 	                                                 task-1-boil  (30 variations)
    1: 	                        task-1-change-the-state-of-matter-of  (30 variations)
    2: 	                                               task-1-freeze  (30 variations)
    3: 	                                                 task-1-melt  (30 variations)
    4: 	             task-10-measure-melting-point-(known-substance)  (436 variations)
    5: 	           task-10-measure-melting-point-(unknown-substance)  (300 variations)
    6: 	                                     task-10-use-thermometer  (540 variations)
    7: 	                                      task-2-power-component  (20 variations)
    8: 	   task-2-power-component-(renewable-vs-nonrenewable-energy)  (20 variations)
    9: 	                                   task-2a-test-conductivity  (900 variations)
   10: 	             task-2a-test-conductivity-of-unknown-substances  (600 variations)
   11: 	                                          task-3-find-animal  (300 variations)
   12: 	                                    task-3-find-living-thing  (300 variations)
   13: 	                                task-3-find-non-living-thing  (300 variations)
   14: 	                                           task-3-find-plant  (300 variations)
   15: 	                                           task-4-grow-fruit  (126 variations)
   16: 	                                           task-4-grow-plant  (126 variations)
   17: 	                                        task-5-chemistry-mix  (32 variations)
   18: 	                task-5-chemistry-mix-paint-(secondary-color)  (36 variations)
   19: 	                 task-5-chemistry-mix-paint-(tertiary-color)  (36 variations)
   20: 	                             task-6-lifespan-(longest-lived)  (125 variations)
   21: 	         task-6-lifespan-(longest-lived-then-shortest-lived)  (125 variations)
   22: 	                            task-6-lifespan-(shortest-lived)  (125 variations)
   23: 	                               task-7-identify-life-stages-1  (14 variations)
   24: 	                               task-7-identify-life-stages-2  (10 variations)
   25: 	                       task-8-inclined-plane-determine-angle  (168 variations)
   26: 	             task-8-inclined-plane-friction-(named-surfaces)  (1386 variations)
   27: 	           task-8-inclined-plane-friction-(unnamed-surfaces)  (162 variations)
   28: 	                    task-9-mendellian-genetics-(known-plant)  (120 variations)
   29: 	                  task-9-mendellian-genetics-(unknown-plant)  (480 variations)
```

## Citing

If this KG-A2C agent is helpful in your work, please cite the following:

Bibtex
```
@misc{scienceworld2022,
    title={ScienceWorld: Is your Agent Smarter than a 5th Grader?},
    author={Ruoyao Wang and Peter Jansen and Marc-Alexandre C{\^o}t{\'e} and Prithviraj Ammanabrolu},
    year={2022},
    eprint={2203.07540},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2203.07540}

@inproceedings{
    ammanabrolu2020graph,
    title={Graph Constrained Reinforcement Learning for Natural Language Action Spaces},
    author={Prithviraj Ammanabrolu and Matthew Hausknecht},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=B1x6w0EtwH}
}
```