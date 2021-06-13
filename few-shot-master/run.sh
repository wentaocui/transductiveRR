for model in transductProtoNet transductMatchingNet transductRelationNet
do
  for shot in 1 5
  do
    python RR_save_features.py --dataset miniImagenet --model ResNet18 --method $model --train_n_way 5 --test_n_way 5 --n_shot $shot --transduct_mode off
    python RR_test.py          --dataset miniImagenet --model ResNet18 --method $model --train_n_way 5 --test_n_way 5 --n_shot $shot --transduct_mode off
    python RR_test.py          --dataset miniImagenet --model ResNet18 --method $model --train_n_way 5 --test_n_way 5 --n_shot $shot --transduct_mode RR
    python RR_save_features.py --dataset miniImagenet --model ResNet18 --method $model --train_n_way 5 --test_n_way 5 --n_shot $shot --transduct_mode MT
    python RR_test.py          --dataset miniImagenet --model ResNet18 --method $model --train_n_way 5 --test_n_way 5 --n_shot $shot --transduct_mode MT
  done
done
