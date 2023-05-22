Task: Predict confliction probability between eNBs. Assuming UEs attached to each eNB has same distribution (implemented with NS3 OnOff Application)

Input: (Number of attached UE of each eNB, Mean of On time, Mean of Off time) $\times$ N, where N is number of eNBs.

Output: Confliction probabilities of combinations of eNBs

Data: Included and located under data/folder. `.npy` data file could be generated with `data_gen.py` script.

Overwrite the `--train_folder` and `--val_folder` to the absolute path of data/train and data/test respectively; Overwrite the `--checkpoint_dir` to the absolute path of the folder to store model parameters.

Start training with:

```python
cat args.txt | xargs python main.py
```

Model parameters would be stored in that folder

Test with:
```python
python plot.py
```