## Explaining the Effects of Clouds on Remote Sensing Scene Classification

This repository comes a along with the paper <i>Explaining the Effects of Clouds on Remote Sensing Scene Classification</i> published in the <i>IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing</i> (<b><i>add link when online</i></b>). 

<img src="https://user-images.githubusercontent.com/77287533/196392412-fb0197af-b61c-4707-a757-52c7ad94ae8d.png" width=80% >

### Prerequisites
This repository has been tested under ```Python 3.8.8``` in a *unix* development environment. <br> 
For a setup, clone the repository and ``cd``to the root of it. <br>
Create a new environment and activate it, for instance, on unix via
```
python -m venv venv && source venv/bin/activate
```
Then install the needed packages via:
```
pip install --upgrade pip
pip install -r requirements.txt
```
The repository builds up on pre-trained neural networks for land cover classification, trained on the cloud-free version of the <a href='https://github.com/schmitt-muc/SEN12MS'>SEN12MS data set</a>. The training pipline and pre-trained networks are available here: <i><a href='https://github.com/schmitt-muc/SEN12MS'>https://github.com/schmitt-muc/SEN12MS</a></i>

### Content and Usage
The repository contains the following components to evaluate Neural Networks trained for <a href='https://github.com/schmitt-muc/SEN12MS'>classification on the SEN12MS data set</a>:
* Computation of the <b>sample-wise cloud-coverage</b> for samples from <a href='https://patricktum.github.io/cloud_removal/'>SEN12MS-CR</a>.
* Computation of <b>classification performances on cloudy and clear samples </b> from <a href='https://patricktum.github.io/cloud_removal/'>SEN12MS-CR</a> and <a href='https://github.com/schmitt-muc/SEN12MS'>SEN12MS</a>.
* Evaluation of the <b>classification performance for different levels of cloud coverage</b>.
* Evaluations of the capability to <b>detect high cloud coverage based on the network output</b>.
* Evaluation and Visualization of <b>saliency maps generated with GradCam for clear and cloudy images</b>.

#### Compute Cloud Coverage
The cloud coverage of the samples in the SEN12MSCR data set can be done by:
```
python ./DataPreparation/compute_cloud_coverage.py --data_root_path ./ROOT/OF/SEN12MSCR \
                                                   [--save_pkl_path './pkl_files/cloud_coverage.pkl']
```
The results are stored in a pickle file given as ``--save_pkl_path`` argument. 

#### Compute class-wise and cloud-coverage clustered band statistics
The band statistics are computed class-wise over a given data split. The pickle files containing the label information and the data split for the cloudy subset of the original test set can be found in ``./DataHandling/DataSplits``, which is also set as a default parameter. <br>
The band statistics can be computed in the following way:
```
python ./DataPreparation/compute_band_statistics.py --data_root_path ./ROOT/OF/SEN12MSCR \
                                                    [--label_split_dir './DataHandling/DataSplits' \
                                                    --cloud_cover_pkl './pkl_files/cloud_coverage.pkl' \
                                                    --save_pkl_path './pkl_files/band_stats.pkl' ]
```
The file passed in ``--cloud_cover_pkl`` is the output of ``./DataPreparation/compute_cloud_coverage.py`` and the results are again stored in a pickle file given as ``--save_pkl_path`` argument. 
#### Evaluate network performance on co-registered cloudy and non-cloudy samples
To evluate the performance of a network on cloudy and corresponding non-cloudy samples call the following script: 
```
python ./DataPreparation/evaluate_network_performance.py --data_root_path ./ROOT/OF/SEN12MSCR\
                                                         --model_type MODEL_TYPE \
                                                         --checkpoint_pth ./PATH/TO/MODEL/CHECKPOINT \
                                                         [--label_split_dir './DataHandling/DataSplits/' \
                                                         --save_folder './pkl_files/']
```
The possible values of ``MODEL_TYPE`` are ["VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201"]. The checkpoint should be a model checkpoint created with the classification training pipeline of the original  <a href='https://github.com/schmitt-muc/SEN12MS'>SEN12MS repository</a> (the training needs to be run on all 13 bands).<br><br>
The results are saved in three pickle files containing statistics for the performance on the clear data, the cloudy data and statistics on the separability of cloudy and non-cloudy samples based on the network's logit output. 

### Visualize Statistics and Model Performances
Class Distribution and Distribution Cloud Coverage
```
python ./DataVisualization/class_distribution.py --data_root_path ./ROOT/OF/SEN12MSCR \
                                                 [--label_split_dir './DataHandling/DataSplits/' \
                                                 --cloud_cover_pkl './pkl_files/cloud_coverage.pkl' \
                                                 --target_folder './ResultPlots/ClassDistributions/]
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/77287533/202240623-155db2ff-7923-4b62-99ad-6bde81371e19.png" width=40% >
  &nbsp;&nbsp;&nbsp;
  <img src="https://user-images.githubusercontent.com/77287533/202240615-adf53185-4570-4e3d-ae67-6095c0b10968.png" width=40% >
</p>

#### Band-wise spectral fingerprint
```
python ./DataVisualization/band_statistics.py [--band_stats_pkl_file './pkl_files/band_stats.pkl' \
                                              --target_folder './ResultPlots/BandStatistics']
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/77287533/202240423-29d58f53-1502-4ca2-b750-11698e29693f.png" width=40% >
  &nbsp;&nbsp;&nbsp;
  <img src="https://user-images.githubusercontent.com/77287533/202240413-6247a98e-2fea-4a71-b88b-f8f47a3c8b53.png" width=40% >
</p>

#### Model Performance vs. Cloud Coverage
```
python ./DataVisualization/performance_vs_cloud_coverage.py --predictions_pkl_path ./PATH/TO/PREDICTION/PICKLE/FILE \
                                                            [--cloud_coverage_pkl_path './pkl_files/cloud_coverage.pkl' \
                                                            --target_folder './ResultPlots/CloudyPerformance']
```
The parameter ``predictions_pkl_path`` should reference to the predictions pickle file created by ``./DataPreparation/evaluate_network_performance.py``.
<p align="center">
  <img src="https://user-images.githubusercontent.com/77287533/202239957-2308c388-326d-4e10-a67d-eb773c6bacee.png" width=40% >
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/77287533/202240241-c10bcd88-4e5e-43f3-9570-4a0d0d39155b.png" width=40% >
  &nbsp;&nbsp;&nbsp;
  <img src="https://user-images.githubusercontent.com/77287533/202240227-c59d4000-1276-4400-a98b-e9773441ca62.png" width=40% >
</p>

#### GradCam application on cloudy and corresponding non-cloudy samples
For the application of the GradCam approach we make use of the interpretability library <a href="https://captum.ai/"> Captum </a> implemented for PyTorch. In order to plot the saliency maps generated with GradCam call the following script:
```
python ./DataVisualization/run_and_plot_grad_cam.py ```
python ./DataVisualization/run_and_plot_grad_cam.py --data_root_path ./ROOT/OF/SEN12MSCR\
                                                    --model_type MODEL_TYPE \
                                                    --checkpoint_pth ./PATH/TO/MODEL/CHECKPOINT \--model_type MODELTYPE \
                                                    --predictions_pkl_path ./PATH/TO/PREDICTION/PICKLE/FILE \
                                                    [--label_split_dir './DataHandling/DataSplits' \
                                                    --cloud_coverage_pkl_path './pkl_files/cloud_coverage.pkl' \
                                                    --target_folder './ResultPlots/grad_cam/saliency_and_pred' \
                                                    --num_eval -1 \
                                                    --num_print 100]
```
The parameter ``num_eval=-1`` means, that the whole data set is evaluated while only the first 100 examples are printed (``num_print=100``). The parameter ``predictions_pkl_path`` should reference to the predictions pickle file created by ``./DataPreparation/evaluate_network_performance.py``.
<p align="center">
  <img src="https://user-images.githubusercontent.com/77287533/202239159-4f6d79ab-c635-4f34-b463-f94a2b7ba5f3.png" width=40% >
  &nbsp;&nbsp;&nbsp;
  <img src="https://user-images.githubusercontent.com/77287533/202239154-1de06720-d831-4164-b0cb-aa0285df31f0.png" width=40% >
</p>


#### Visualization of GradCam statistics
To visualize the GradCam statistics violin plots call the script `PlotGradCamViolinPlots.py` with the saliency statistic pickle files:
```
python ./DataVisualization/grad_cam_violin_plots.py  --stats_clear_path ./PATH/TO/CLEAR/STATS/PKL/FILE \
                                                     --stats_cloudy_path ./PATH/TO/CLOUDY/STATS/PKL/FILE \
                                                     --save_path './ResultPlots/grad_cam/Statistics/']
```
If stats_clear or stats_cloudy is not given, only the plots for the given pkl-files are generated. 
<p align="center">
<img src="https://user-images.githubusercontent.com/77287533/202238003-fa9981af-8e02-4143-9591-8fdcc522aaa8.png" width=40% >
</p>

### Citation
If you find our code or results useful for your research, please consider citing: 
```
@article{gawlikowski2022explaining,
  title={Explaining the Effects of Clouds on Remote Sensing Scene Classification},
  author={Gawlikowski, Jakob and Ebel, Patrick and Schmitt, Michael and Zhu, Xiao Xiang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2022}
}
```

