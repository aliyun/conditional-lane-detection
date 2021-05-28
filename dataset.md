# Datasets
We conducted experiments on CurveLanes, CULane and TuSimple. The settings of these datasets are as follows. 
Note, [your-data-path] is the path you specifiying to save the datasets and [project-root] is the root path of this project.

## CurveLanes
[\[Website\]](https://github.com/SoulmateB/CurveLanes)

```bash
mkdir datasets # if it does not already exists
cd datasets
gdown "https://drive.google.com/uc?id=1nTB2Cdyd0cY3nVB1rZ6Z00YjhKLvzIqr"
gdown "https://drive.google.com/uc?id=1iv-2Z9B6cfncogRhFPHKqNlt-u7hQnZd"
gdown "https://drive.google.com/uc?id=1n2sFDdy2KAaw-7siO7HWuwxUeVb6SXfN"
gdown "https://drive.google.com/uc?id=1xiz2oD4A0rlt3TGFdz5uzU1s-a0SbsX8"
gdown "https://drive.google.com/uc?id=1vpFSytqlsJA-rzfuY2lyXmZvEKpaovjX"
gdown "https://drive.google.com/uc?id=1NZLvaBWj0Mnuo07bxKT7shxqi9upSegJ"
unrar x Curvelanes.part1.rar

# We convert to CULane formal using prepare_curvelanes_datasets.py script
cd [project-root]
python tools/condlanenet/curvelanes/prepare_curvelanes_datasets.py [your-data-path/Curvelanes]
```

Then the directory should be like follows:
```
[your-data-path]/Curvelanes
├── test
│   └── images
├── train
│   └── images
│   └── labels
│   └── train.txt
└── valid
    └── images
    └── labels
    └── valid.txt

```


## CULane
[\[Website\]](https://xingangpan.github.io/projects/CULane.html)

Inside [your-data-path], run the following:
```bash
mkdir datasets # if it does not already exists
cd datasets
mkdir culane
# train & validation images (~30 GB)
gdown "https://drive.google.com/uc?id=1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk"
gdown "https://drive.google.com/uc?id=1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL"
gdown "https://drive.google.com/uc?id=14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL"
tar xf driver_23_30frame.tar.gz
tar xf driver_161_90frame.tar.gz
tar xf driver_182_30frame.tar.gz
# test images (~10 GB)
gdown "https://drive.google.com/uc?id=1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8"
gdown "https://drive.google.com/uc?id=1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV"
gdown "https://drive.google.com/uc?id=1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7"
tar xf driver_37_30frame.tar.gz
tar xf driver_100_30frame.tar.gz
tar xf driver_193_90frame.tar.gzt
# all annotations (train, val and test)
gdown "https://drive.google.com/uc?id=1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu"
tar xf annotations_new.tar.gz
gdown "https://drive.google.com/uc?id=18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd"
tar xf list.tar.gz
```

Then the directory should be like follows:
```
[your-data-path]/culane
├── driver_23_30frame
├── driver_37_30frame
├── driver_100_30frame
├── driver_161_90frame
├── driver_182_30frame
├── driver_193_90frame
└── list
    └── test_split
    |   ├── test0_normal.txt
    |   ├── test1_crowd.txt
    |   ├── test2_hlight.txt
    |   ├── test3_shadow.txt
    |   ├── test4_noline.txt
    |   ├── test5_arrow.txt
    |   ├── test6_curve.txt
    |   ├── test7_cross.txt
    |   └── test8_night.txt
    └── train.txt
    └── test.txt
    └── val.txt

```

## TuSimple
[\[Website\]](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection)

Inside [your path], run the following:
```bash
mkdir datasets # if it does not already exists
cd datasets
# train & validation data (~10 GB)
mkdir tusimple
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip"
unzip train_set.zip -d tusimple
# test images (~10 GB)
mkdir tusimple-test
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip"
unzip test_set.zip -d tusimple-test
# test annotations
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json" -P tusimple-test/
cd ..
```

Then the directory should be like follows:
```
[your-data-path]/tusimple
├── clips
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
├── label_data_0601.json
├── test_label.json
└── test_baseline.json

```
