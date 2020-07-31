## Feature Generation and Preprocessing (In particular Order)

1. src/scripts/preprocess/shortening_atma.py
2. src/scripts/preprocess/video_list_generator.py
3. src/scripts/preprocess/Annotation_list_generator.py
4. src/scripts/preprocess/generate_audio_files.py
5. src/scripts/feature_generation/audio_feature_generation/vgg/gen_audio_set_feats.py
6. src/scripts/feature_generation/video_feature_generation/c3d/feature_extractor_vid.py
7. src/scripts/preprocess/make_audio_h5py.py 
8. src/scripts/preprocess/generate_train_test_path.py
9. src/scripts/preprocess/parse_annotation_file.py
10. src/scripts/preprocess/generate_time_duration_file.py
11. src/scripts/preprocess/generate_no_label.py
12. src/scripts/preprocess/generate_splitlist.py
13. src/scripts/preprocess/generate_h5py.py 
14. src/scripts/preprocess/generate_h5pyWithCaption.py

## Training and Localization

1. src/scripts/train/main.py

## Changing Audio Features (In Order)

1. Generate appropriate audio features from src/scripts/feature_generation/ 
2. Change AUDIO_FEATURES_TO_USE variable in src/scripts/constants.py
3. src/scripts/preprocess/make_audio_h5py.py
4. src/scripts/preprocess/generate_no_label.py
5. src/scripts/preprocess/generate_h5py.py
6. src/scripts/preprocess/generate_h5pyWithCaption.py
