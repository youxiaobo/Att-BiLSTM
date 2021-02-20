# Att-BiLSTM

## step1: model train

### BiLSTM

```
train command: python main.py train --train_data_root='./data/data_20200510/train_feat2/50_20/'
model path: ./checkpoints/20200601/lr0.001_50_20_0_best_checkpoints.pth
```

### Att-BiLSTM

```
train command: python main_attHyper.py train --train_data_root='./data/data_20200510/train_feat2/50_20'
model path: ./checkpoints/20200602/attwin[1]train_feat2_0_best_checkpoints.pth
```



## step2: simulated data test

### BiLSTM

```
test command: python main.py test --test_data_root='./data/data_20200510/test_originFeat2/' --test_result_root='./data/data_20200510/test/' --batch_size=1 --load_model_path='./checkpoints/20200601/lr0.001_50_20_0_best_checkpoints.pth'
```



### Att-BiLSTM

```
test command: python main.py test --test_data_root='./data/data_20200510/test_originFeat2/' --test_result_root='./data/data_20200510/test/' --batch_size=1 --load_model_path='./checkpoints/20200602/attwin[1]train_feat2_0_best_checkpoints.pth' --attention_win=1
```



## step3: in-house data test

```
test command:
python main.py test --test_data_root='./data/200424 90+92+0 exp200 int0 009good_originFeat4/' --test_result_root='./data/200424 90+92+0 exp200 int0 009good/' --batch_size=1 --load_model_path='./checkpoints/20200520_1/50_20_0_best_checkpoints.pth'
python main.py test --test_data_root='./data/200428 91bfpkdel 20mchrab7 exp200ms int2s 009 good_originFeat4/' --test_result_root='./data/200428 91bfpkdel 20mchrab7 exp200ms int2s 009 good/' --batch_size=1 --load_model_path='./checkpoints/20200520_1/50_20_0_best_checkpoints.pth'
python main.py test --test_data_root='./data/200502  105 93gfpsec61btf lyso647 exp200ms in2s 057_originFeat4/' --test_result_root='./data/ 200502  105 93gfpsec61btf lyso647 exp200ms in2s 057/' --batch_size=1 --load_model_path='./checkpoints/20200520_1/50_20_0_best_checkpoints.pth'
```

