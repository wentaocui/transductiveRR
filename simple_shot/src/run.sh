
for backbone in resnet18 wideres
do
  for dataset in miniImagenet tiered_imagenet cub
  do
    for shot in 1 5
    do
      python ./src/train.py --data-for-all $dataset --arch-for-all $backbone --meta-val-way 5 --eval-shot $shot
      python ./src/train.py --data-for-all $dataset --arch-for-all $backbone --meta-val-way 5 --eval-shot $shot  --enable-transduct
    done
  done
done
