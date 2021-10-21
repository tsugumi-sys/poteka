test: test.py
	python test.py

preprocess_pressure_data: dataset/data-making/make_PRS_SLP_image.py
	python dataset/data-making/make_PRS_SLP_image.py \
		--data_root_path=../data \
		--target=prs \
		--n_jobs=5

preprocess_slp_data: dataset/data-making/make_PRS_SLP_image.py
	python dataset/data-making/make_PRS_SLP_image.py \
		--data_root_path=../data \
		--target=slp \
		--n_jobs=5

preprocess_humidity_data: dataset/data-making/make_humidity_image.py
	python dataset/data-making/make_humidity_image.py \
	--data_root_path=../data \
	--n_jobs=10

preprocess_temp_data: dataset/data-making/make_temp_image.py
	python dataset/data-making/make_temp_image.py \
	--data_root_path=../data \
	--n_jobs=5

preprocess_wind_data: dataset/data-making/make_wind_image.py
	python dataset/data-making/make_wind_image.py \
	--data_root_path=../data \
	--n_jobs=10 \
	--target=abs