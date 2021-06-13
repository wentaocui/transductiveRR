python3 -m src.main \
		-F logs/tim_gd/tiered/wideres \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/wideres" \
		dataset.split_dir="split/tiered" \
		model.arch='wideres' \
		model.num_classes=351 \
		tim.iter=600 \
		evaluate=True \
		eval.method='tim_gd' \


python3 -m src.main \
		-F logs/tim_gd/tiered/densenet121 \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/densenet121" \
		dataset.split_dir="split/tiered" \
		dataset.batch_size=16 \
		model.arch='densenet121' \
		model.num_classes=351 \
		tim.iter=600 \
		evaluate=True \
		eval.method='tim_gd' \

