# protoseg

Prototyped Segmentation

## Workflow

![workflow diagram](http://www.plantuml.com/plantuml/proxy?src=https://raw.github.com/chriamue/protoseg/master/res/workflow.puml)

## Data

Images should be copied to ./data folder.
Images for training to train folder, the masks to train_masks folder.
Images for validation to val folder, the masks to val_masks folder.
Images for testing into test folder.

## Config

Every run is stored in a config file like:
[config](configs/gluoncv.yml)

## Backends

There should exist multiple backends.

* gluoncv
* pytorch-semseg

```yml
gluoncv:
  backend: gluoncv_backend
  backbone: resnet50
```

```yml
ptsemseg_unet:
  backend: ptsemseg_backend
  backbone: unet
  classes: 2
  width: 572
  height: 572
  mask_width: 388
  mask_height: 388
  orig_width: 768
  orig_height: 768
  gray_img: True
```

```yml
ptsemseg_segnet:
  backend: ptsemseg_backend
  backbone: segnet
  classes: 2
  width: 512
  height: 512
```

## Report

A report as PDF file of the results can be created.

## Kaggle Competition Data

In folder ./scripts is ultrasound-nerve-segmentation.py which should be run as

```bash
python3 ./scripts/ultrasound-nerve-segmentation.py /path/to/competition-data data/
```

The script extracts competition images and copies them to the data folder.
