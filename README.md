# GA-FER

Code implementation of paper "Neural architecture search using genetic algorithm for facial expression recognition" published on GECCO 2022.

## Requirements
- Pytorch

Torch 1.1.0 or higher and torchvision 0.11.2 or higher are required.

(Example: torch == 1.10.1,  torchvision  == 0.11.2,  numpy == 1.19.5)

## How to run

**Setp 1. Data Preparation**

  Download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), and make sure it have a structure like following:
 
```
- datasets/raf-basic/
         EmoLabel/
             list_patition_label.txt
	     new_10_noise.txt
	     new_20_noise.txt
         Image/aligned/
	     train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...
```


**Step 2. Then you need to change the path where the dataset is loaded to your dataset path.
          The changes can be found on lines 47 and 48 of the cifar10.py file in the template folder.**

```
  trainloader = data_loader.get_train_loader('/home/dengshuchao/datasets/RafDb/raf-basic/',64,1,True,True)
  validloader = data_loader.get_valid_loader('/home/dengshuchao/datasets/RafDb/raf-basic/',64,1,False,True)
```

**Step 3. Set hyperparameters in *global.ini*.**

Ensure the following status before running：

[evolution_status]

is_running = 0

**Step 4. Run `python GA-FER-evolve.py`  or `nohup python -u GA-FER-evolve.py > GA-FER-evolve.log 2>&1 &`**

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

