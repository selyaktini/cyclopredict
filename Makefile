setup:
	pip install -r requirements.txt

run:
	python src/01_get_data.py
	python src/02_preprocess.py
	python src/04_train_mlp.py
	python src/05_plot_results.py

clean:
	rm -rf data/processed/* models/*.joblib
