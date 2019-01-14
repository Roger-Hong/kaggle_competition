gcloud ml-engine jobs submit training minst_cnn_35 \
	--job-dir gs://federer-hyj2721-kaggle-minst/trainer/output/ \
	--module-name trainer.task \
	--package-path trainer/ \
	--region us-west1 \
	--config=trainer/cloudml-gpu.yaml \
	-- \
    --layer_nodes=32_64_64_64_64 \
    --train_rounds=200 \
    --using_gpu=False
