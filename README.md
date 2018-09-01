# protoseg

Prototyped Segmentation

## Data

Images should be copied to ./data folder.
Images for training to train folder, the masks to train_masks folder.
Images for validation to val folder, the masks to val_masks folder.
Images for testing into test folder.

## Kaggle Competition Data

In folder ./scripts is ultrasound-nerve-segmentation.py which should be run as

```bash
python3 ./scripts/ultrasound-nerve-segmentation.py /path/to/competition-data data/
```

The script extracts competition images and copies them to the data folder.