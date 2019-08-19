# MTPAD

This is the inference code for an automatic iris presentation attack detection (PAD) solution. It is trained based on the darknet framework, which can simultaneously detect the iris region and produce the presentation attack score. It computes the score for individual iris images and produces the APCER, BPCER, and ROC metrics. It has been intended to use as a baseline so that other PAD solutions can be compared against. 

Some of the detection results have been visualized here. 


![image](https://github.com/cunjian/MTPAD/blob/master/results/live.png "Logo Title Text 1")

![image](https://github.com/cunjian/MTPAD/blob/master/results/contact.png "Logo Title Text 1")

![image](https://github.com/cunjian/MTPAD/blob/master/results/print.png "Logo Title Text 1")

![image](https://github.com/cunjian/MTPAD/blob/master/results/artifical_eye.png "Logo Title Text 1")

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{WACVW2018,
  title={A Multi-Task Convolutional Neural Network for Joint Iris Detection and Presentation Attack Detection},
  author={Chen, Cunjian and Ross, Arun},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV) Workshop},
  year={2018}
}
