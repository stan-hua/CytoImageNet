# CytoImageNet: A large-scale pretraining dataset for bioimage transfer learning

![cytoimagenet_plot](https://github.com/stan-hua/CytoImageNet/blob/c65f28d6ca27cb7ecb37dfa442576083b5496522/figures/github_images/cytoimagenet_plot.png)

**Paper**: https://arxiv.org/abs/2111.11646

**Dataset**: https://www.kaggle.com/stanleyhua/cytoimagenet

**Recorded talk:** https://youtu.be/qfEA-UR6tVM

Motivation
---
In past few decades, the increase in speed of data collection has led to the dawn of so-called '*big data*'. In the field of molecular biology, 
this was seen in 'high throughput sequencing', where DNA and protein assays exceeded the capability of scientists to analyze large amount of datas.
The need to develop computational methods that can match the rate at which data is collected sparked the era of 'bioinformatics'. In more recent years,
our ability to capture and store biological images has grown tremendously to the point where we may now consider microscopy images '*big data*'. 

Thus, a need for **automated methods to help analyze biological images** emerges.
Here, we take inspiration from the success of ImageNet to curate CytoImageNet; a large-scale dataset of weakly labeled microscopy images. We believe that pretraining deep learning models on CytoImageNet will result in models that can extract image features with stronger biological signals from microscopy images, in comparison to ImageNet features that were trained originally on natural images (e.g. buses, airplanes).


## Results

We show that concatenation of CytoImageNet-pretrained and ImageNet-pretrained features yields state-of-the-art results on bioimage transfer tasks. This implies that CytoImageNet and ImageNet pretraining result in the learning of different albeit meaningful image representations.

Interestingly, our trained model only achieved 13.42% accuracy on the training set and 11.32% on the validation set. Yet, it still produced features competitive to ImageNet on all 3 downstream microscopy classification tasks. Given the closer domain of CytoImageNet, we find it surprising that CytoImageNet-pretrained features alone don’t outperform ImageNet-pretrained features.

A recent study on ImageNet pretraining reported a strong correlation between ImageNet validation accuracy and transfer accuracy. It may be that we haven’t had the opportunity to optimize the model enough, and we believe this may be explored in future work. In addition, it has been found that pretraining on a subset of ImageNet (with classes more similar to the target task) can improve transfer performance. Future researchers may explore pretraining on labels from specific categories (e.g. phenotype) if the target task focuses more on cell phenotype, compounds/treatment, or protein localization.

Read more [**here**](https://arxiv.org/abs/2111.11646).

## About the data
**890,737** total images. **894 classes** (~1000 images per class). 

Microscopy images belong to **40 openly available datasets** from the following databases: Recursion, Image Data Resource, Broad Bioimage Benchmark Collection,
Kaggle and the Cell Image Library. See below for the list of datasets included.

The classes are ***soft/weak labels***, so overlap is possible. Labels were assigned based on image metadata provided in the originating datasets. Chosen label could 
correspond to any of [organism, cell_type, cell_visible, phenotype, compound, gene, sirna].

|       Category      |     # of labels    |
|:-------------------:|:------------------:|
|       compound      |         637        |
|       phenotype     |          93        |
|       cell_type     |          44        |
|         gene        |          43        |
|     cell_visible    |          38        |
|         sirna       |          36        |
|       organism      |          3         |

### Metadata associated with each image
> * **label**: Class assigned to image
> * **category**: Name of column where label originates from (e.g. organism)

> * **database**: Database containing source dataset
> * **name**: Name of source dataset (*created if no explicit name in database*)
> * **dir_name**: Shorthand naming convention of dataset (*created if no explicit shorthand in database*)
> * **path**: Relative path to folder containing image file. (e.g. /cytoimagenet/human)
> * **filename**: Standardized filename based on binary of image number in class (e.g. human-00001011.png)
> * **idx**: Index that maps to original image from source dataset. (e.g. bbbbc041-14631)

> * **organism**: Biological organism (e.g. human)
> * **cell_type**: May refer to cell class (e.g. red blood cell) or cell line (e.g. u2os)
> * **cell_visible**: May refer to cell components that were stained (e.g. actin) or features of an organism that were stained for (e.g. pharynx)
> * **phenotype**: May refer to disease condition (e.g. leukemia), mechanism-of-action (MOA), cell cycle stage (e.g. telophase), etc.
> * **compound**: Name of compound treatment
> * **sirna**: Name of siRNA treatment
> * **gene**: Name of gene (or protein) targeted (*" targeted" added to avoid overlapping labels with cell_visible*)

> * **microscopy**: Microscopy modality (e.g. fluorescence)
> * **crop**: True if image is a crop from an image. False, otherwise.
> * **scaling**: Length of crop side relative to size of original image (e.g. 0.5 corresponds to a 0.5 by 0.5 window ~ 1/4 the original image)
> * **x_min, x_max, y_min, y_max**: Slicing indices to create crop from original image

**NOTE**: In the case of multi-labels in each category, possible labels are separated by a "|" (e.g. nucleus|actin|mitochondria).

**EXTRA NOTE**: All labels were converted to lowercase, which may make searching labels difficult, particularly with compound labels.

### Availability of Data
CytoImageNet is now available on Kaggle: https://www.kaggle.com/stanleyhua/cytoimagenet (~56 GB).

---
# Methods

## Data Cleaning

### Annotation
65 datasets were manually searched one by one, requiring dataset-specific annotation processing due to inconsistency and sometimes unreliability of available metadata. If metadata was available for all images, columns were selected and often transformed to create the standardized metadata for each image above. If not, metadata was curated based on available information about the dataset and assigned to all images in the dataset. Labels found in other datasets with different names were standardized and merged if found (e.g. neutrophil -> white blood cell, GPCR -> cell membrane). In the case that multiple labels exist for a category, multi-labels are separated by a "|" (e.g. nucleus|actin|mitochondria).

For fluorescent microscopy images, images are typically grayscale with around 1-7+ images depending on what is being stained for. Yet, these correspond to 1 row in the dataset. These pseudo (uncreated) images are given filenames that allow mapping to the original existing images to merge. These images with 1-7+ channels are merged as part of the data processing pipeline, if selected. 

In total, this produced **2.7 million rows** with each row corresponding to an image and a unique image index.

**RELEVANT CODE**: [`clean_metadata.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/data_curation/clean_metadata.py), [`describe_dataset.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/data_curation/describe_dataset.py)

---

### Weak Label Assignment & Stratified Downsampling
Of the 2.7 million row table, each column from [organism, cell_type, cell_visible, phenotype, compound, gene, sirna] were searched for unique values and their counts, ignoring how labels group together between columns. To create near to 1000 labels, potential labels with counts equal to or above a chosen threshold of 287 served as *potential labels*. Beginning from the least counts, potential labels were iterated through, keeping track of rows that were already assigned labels in a hash table via their unique index. Stratified sampling based on metadata curated is used to improve diversity of images selected for labels.

***Pseudo-Code***: for each potential label
1. Filter for images containing potential label in metadata AND not currently in hash table.
2. Skip if < 287 images for potential label. End iteration.
3. If less than 1000 images, skip to step 6.
4. If > 10,000 images, sample 10,000 rows by a preliminary stratified sampling on columns in [dataset, organism and cell type].
5. If > 1000 images, sample 1000 rows by stratified sampling on columns in [organism, cell_type, cell visible, sirna, compound, phenotype].
6. Save potential label and update hash table with used files.

**RELEVANT CODE**: [`analyze_metadata.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/data_curation/analyze_metadata.py)

---

### Image Data Cleaning & Standardization
In general, there is no one-size-fits-all when it comes to microscopy images since the types of images collected vary based on the study. And a lack of a golden standard for storing image data makes data cleaning a dataset-specific task. The following steps are done on selected images...

* **Standardize file format** to PNG from other formats (TIF, JPG, FLEX, BMP, etc.)
* Converting RGB images **to grayscale**.
* If merging fluorescently stained channels, **normalize** each channel using 0.1th and 99.9th percentile pixel intensity, then **merge** channels to create grayscale images. 

**NOTE**: Brightfield microscopy images are separated from fluorescent microscopy images.

**NOTE**: The dataset contains single channel images (e.g. an image only stained for actin, brightfield images).

**EXTRA NOTE**: Normalization procedure follows preprocessing used in training [DeepLoc](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5408780/) and helps in brightening dim images.

> **Merging Procedure**
![channel merging](https://user-images.githubusercontent.com/63123494/130717702-184e3b14-f4ad-4e27-b26b-a1d95e11c6e3.png)

**RELEVANT CODE**: [`preprocessor.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/data_processing/preprocessor.py), [`prepare_dataset.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/data_processing/prepare_dataset.py)

---

### Upsampling Classes
![upsampling_procedure](https://user-images.githubusercontent.com/63123494/130723750-c7400d1a-31e1-4393-9ac3-e6f7117a58d7.png)

To increase image diversity and class size, we take 1-4 crops per image of different scale, preferably. Since microscopy images are not center-focused, we split the image into 4 non-overlapping windows. For each window, we randomly choose the crop size based on a scaling factor from [1/2, 1/4, 1/8, 1/16], making sure that crop sizes are above 70x70 pixels. Then we take the random crop from anywhere in the window. Crops are filtered for artifacts (completely white/black, too dark). Each crop is normalized once again with respect to the 0.1th and 99.9th percentile pixel intensity.

We extract **ImageNet features** and use **UMAPs** (a dimensionality reduction method) to visualize the effects of our chosen upsampling method on 20 randomly chosen classes. After upsampling, we increase the diversity of our classes by introducing different resolutions (scaling). Notice that it becomes more difficult for ImageNet features to separate images from different classes.

![upsampling_effects](https://user-images.githubusercontent.com/63123494/130719806-e36fe929-f4b0-49de-b19d-3a5203e71851.png)

**RELEVANT CODE**: [`prepare_dataset.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/data_processing/prepare_dataset.py), [`feature_extraction.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/feature_extraction.py), [`visualize_classes.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/visualize_classes.py)

---

### Quality Control
*Pre-upsampling*, we discard PIL **unreadable** images from potential images.

*Post-upsampling*, we discard the following kinds of images:
1. **Uniform**/constant images
> * This filter removes images with 0.1th and 99.9th percentile pixel intensities.
2. **Binary masks**
> * This filter removes images with only two unique pixel intensities.
3. **Dim**/empty images
> * This filter removes images whose 75th percentile pixel intensity is equal to 0. Intuitively, this would suggest that most of the image is dark. '75th percentile' was chosen based on plotting examples of dim images and experimenting with different thresholds.

> **NOTE**: We have no guarantees for the quality of the data outside of these quality checks.

**RELEVANT CODE**: [`prepare_dataset.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/data_processing/prepare_dataset.py)

---

## Model Training

CytoImageNet is split into a training and validation set with 10% used for validation. This yields roughly 900 training samples for each label. Images are fed in batches of 64 with random 0 to 360 degrees rotations. We train convolutional networks (EfficientNetB0) to classify one of the 894 labels, by minimizing the categorical cross-entropy loss of predictions to ground truth labels. Randomly-initialized models were trained for 24 epochs (2 weeks) on an NVIDIA Tesla K40C. The loss was optimized via the Adam optimizer with learning rate of 0.001

**RELEVANT CODE**: [`model_pretraining.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/model_pretraining.py)

## Evaluation (Transfer Tasks)
We validate the performance of our CytoImageNet features on three classification-based transfer tasks: (1) **BBBC021 evaluation protocol** from the Broad Institute, (2) the **Cells Out of Sample (COOS-7)** dataset, and (3) the **CyCLOPS Wt2** dataset.

#### Methods of Feature Extraction
Since ImageNet does not contain microscopy images, we extract image features in 4 different methods to create a fairer comparison:
> 1. **concatenation** and **normalization**
>    * normalize each channel filling in [0,1] with the 0.1th and 99.9th percentile pixel intensity
>    * extract features from each channel and concatenate, resulting in 1280 x (n channels) features
> 2. **concatenation** and no normalization
>    * extract features from each channel and concatenate, resulting in 1280 x (n channels) features
> 3. **merge** and **normalization**
>    * normalize each channel filling in [0,1] with the 0.1th and 99.9th percentile pixel intensity
>    * merge channel images into 1 grayscale image then extract features, resulting in 1280 features
> 4. **merge** and no normalization
>    * merge channel images into 1 grayscale image then extract features, resulting in 1280 features

---

### BBBC021 Evaluation Protocol
The procedure is as follows:
1. Extract image features from ~2000 images (*each 'image' is made of **3** grayscale fluorescent microscopy images*).
2. Aggregate mean feature vector on treatment (compound - at specific concentration). Resulting in 103 feature vectors corresponding to 103 treatments.
3. Using **1-nearest neighbors** (kNN), classify mechanism-of-action (MOA) label, excluding neighbors with same compound treatments.
4. Report accuracy, termed *'not-same-compound' **(NSC) accuracy***.

### COOS-7
A dataset of single-cell mouse cells, COOS-7 was originally designed to test the out-of-sample generalizability of trained classifiers. For each of the 4 test sets, the evaluation procedure is as follows:
1. Extract image features (*each 'image' is made of **2** grayscale fluorescent microscopy images*)
2. Using 11-nearest neighbors trained on features extracted from the training set, classify the protein's localization given in one of 7 labels.


### CyCLOPS
This dataset is composed of single-cell images of yeast cells. The evaluation procedure is as follows:
1. Extract image features (*each 'image' is made of **2** grayscale fluorescent microscopy images*)
2. Using 11-nearest neighbors, classify the protein's localization given in one of 17 labels.

**RELEVANT CODE**: [`model_evaluation.py`](https://github.com/stan-hua/CytoImageNet/blob/master/scripts/model_evaluation.py)

---

## Sources of Data

**Database**|**Number of Labels Contributed**
-----|-----
Recursion|651
Image Data Resource|450
Broad Bioimage Benchmark Collection|202
Kaggle|27
Cell Image Library|1

CytoImageNet image data comes from the open-source datasets listed below.
> **NOTE**: If dataset name is too long (e.g. name of source publication), a shorter name is given.

### Broad Bioimage Benchmark Collection
> * [C. elegans infection marker](https://bbbc.broadinstitute.org/BBBC012)
> * [C. elegans live/dead assay](https://bbbc.broadinstitute.org/BBBC010)
> * [C. elegans metabolism assay](https://bbbc.broadinstitute.org/BBBC011)
> * [Cell Cycle Jurkat Cells ](https://bbbc.broadinstitute.org/BBBC048)
> * [Human HT29 colon-cancer cells shRNAi screen](https://bbbc.broadinstitute.org/BBBC017)
> * [Human Hepatocyte and Murine Fibroblast cells – Co-culture experiment](https://bbbc.broadinstitute.org/BBBC026)
> * [Human U2OS cells (out of focus)](https://bbbc.broadinstitute.org/BBBC006)
> * [Human U2OS cells - compound cell-painting experiment](https://bbbc.broadinstitute.org/BBBC022)
> * [Human U2OS cells cytoplasm–nucleus translocation](https://bbbc.broadinstitute.org/BBBC013)
> * [Human U2OS cells cytoplasm–nucleus translocation (2)](https://bbbc.broadinstitute.org/BBBC014)
> * [Human U2OS cells transfluor](https://bbbc.broadinstitute.org/BBBC015)
> * [Human U2OS cells – RNAi Cell Painting experiment](https://bbbc.broadinstitute.org/BBBC025)
> * [Human White Blood Cells](https://bbbc.broadinstitute.org/BBBC045)
> * [Human kidney cortex cells](https://bbbc.broadinstitute.org/BBBC051)
> * [Kaggle 2018 Data Science Bowl](https://bbbc.broadinstitute.org/BBBC038)
> * [Murine bone-marrow derived macrophages](https://bbbc.broadinstitute.org/BBBC020)
> * [P. vivax (malaria) infected human blood smears](https://bbbc.broadinstitute.org/BBBC041)
> * [Synthetic cells](https://bbbc.broadinstitute.org/BBBC005)

### Cell Image Library
> * [Kinome Atlas](http://cellimagelibrary.org/pages/kinome_atlas)

### Image Data Resource
> * [Adenovirus](https://idr.openmicroscopy.org/webclient/?show=screen-2406)
> * [Chemical-Genetic Interaction Map](https://idr.openmicroscopy.org/webclient/?show=screen-1151)
> * [Compound Profiling](https://idr.openmicroscopy.org/webclient/?show=screen-1251)
> * [Early Secretory Pathway](https://idr.openmicroscopy.org/webclient/?show=screen-803)
> * [Mitotic Atlas](https://idr.openmicroscopy.org/webclient/?show=project-404)
> * [Pericentriolar Material](https://idr.openmicroscopy.org/webclient/?show=project-51)
> * [Perturbation](https://idr.openmicroscopy.org/webclient/?show=screen-2701)
> * [Phenomic Profiling](https://idr.openmicroscopy.org/webclient/?show=screen-2651)
> * [Plasticity](https://idr.openmicroscopy.org/webclient/?show=screen-51)
> * [Subcellular Localization](https://idr.openmicroscopy.org/webclient/?show=screen-2952)
> * [Variation in Human iPSC lines](https://idr.openmicroscopy.org/webclient/?show=screen-2051)
> * [Yeast Meiosis](https://idr.openmicroscopy.org/webclient/?show=project-904)
> * [siRNA screen for cell size and RNA production](https://idr.openmicroscopy.org/webclient/?show=screen-2751)

### Kaggle
> * [Cell Cycle Experiments](https://www.kaggle.com/paultimothymooney/cell-cycle-experiments)
> * [Human Protein Atlas - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification/data)
> * [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification/data)
> * [Leukemia Classification](https://www.kaggle.com/andrewmvd/leukemia-classification)

### Recursion
> * [RxRx1](https://www.rxrx.ai/rxrx1#the-data)
> * [RxRx19a](https://www.rxrx.ai/rxrx19a)
> * [RxRx19b](https://www.rxrx.ai/rxrx19b)
> * [RxRx2](https://www.rxrx.ai/rxrx2)

---

## Acknowledgements
This project was supervised by Professor [Alan Moses](http://www.moseslab.csb.utoronto.ca/people/amoses/) and Dr. [Alex Lu](https://www.alexluresearch.com/), who are both experts in the field of representation learning for biological images and sequence data. The project was funded by the University of Toronto CSB Undergraduate Research Award. 

Special thanks to Professor [Juan Caicedo](https://www.broadinstitute.org/bios/juan-c-caicedo) of the Broad Institute for his instruction on the BBBC021 evaluation protocol, and Professor [Anne Carpenter](https://www.broadinstitute.org/bios/anne-e-carpenter) for her help early on in understanding datasets in the Broad Bioimage Benchmark Collection.
