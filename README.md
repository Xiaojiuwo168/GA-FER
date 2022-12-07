# GA-FER

Code implementation of GECCO 2022 paper "Neural architecture search using genetic algorithm for facial expression recognition".

## Requirements
- Pytorch

Torch 1.1.0 or higher and torchvision 0.11.2 or higher are required.
(Example: torch == 1.10.1,  torchvision  == 0.11.2,  numpy == 1.19.5)

## How to run

1. Data Preparation

  Download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), and make sure it have a structure like following:
 
```
- datasets/raf-basic/
         EmoLabel/
             list_patition_label.txt
         Image/aligned/
	     train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...
```

2. Set hyperparameters in *global.ini*.

Ensure the following status before running：

[evolution_status]
is_running = 0

3. Run `python GA-FER-evolve.py`.

If you have any questions, please feel free to raise "issues" for discussion.

## Citing GA-FER

It would be greatly appreciated if the following paper can be cited when the code has helped your research.

```
@inproceedings{deng2022neural,
  title={Neural architecture search using genetic algorithm for facial expression recognition},
  author={Deng, Shuchao and Sun, Yanan and Galvan, Edgar},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  pages={423--426},
  year={2022}
}
```

