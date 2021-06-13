
python3 -m src.main \
		-F logs/tim_gd/mini/wideres \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/wideres" \
		dataset.split_dir="split/mini" \
		model.arch='wideres' \
		evaluate=True \
		tim.iter=600 \
		eval.method='tim_gd' \


python3 -m src.main \
		-F logs/tim_gd/mini/densenet121 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/densenet121" \
		dataset.split_dir="split/mini" \
		model.arch='densenet121' \
		evaluate=True \
		tim.iter=600 \
		eval.method='tim_gd' \