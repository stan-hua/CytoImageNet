# CytoImageNet
A pretraining dataset for bioimage transfer learning. 

Motivation
---
In past few decades, the increase in speed of data collection has led to the dawn of so-called '*big data*'. In the field of molecular biology, 
this was seen in 'high throughput sequencing', where DNA and protein assays exceeded the capability of scientists to analyze large amount of datas.
The need to develop computational methods that can match the rate at which data is collected sparked the era of 'bioinformatics'. In more recent years,
our ability to capture and store biological images has grown tremendously to the point where we may now consider microscopy images '*big data*'. 

Thus, a need for **automated methods to help analyze biological images** emerges.
Here, we take inspiration from the success of ImageNet to curate CytoImageNet; a large-scale dataset of weakly labeled microscopy images. We believe that pretraining deep learning models on CytoImageNet will result in models that can extract image features with stronger biological signals, compared to ImageNet features that were trained originally on natural images (e.g. buses, airplanes).

## About the data
**890,217** total images. **894 classes** (~1000 per class).

Microscopy images belong to **40 openly available datasets** from the following databases: Recursion, Image Data Resource, Broad Bioimage Benchmark Collection,
Kaggle and the Cell Image Library.
![database_composition](https://user-images.githubusercontent.com/63123494/130711398-fcd9d10b-9162-4284-b294-76be30b8a61b.png)

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
> * **filename**: Standardized filename based on binary of image number in class (e.g. human-00001011)
> * **idx**: Index that maps to original image from source dataset. (e.g. bbbbc041-14631)

> * **organism**: Biological organism (e.g. human)
> * **cell_type**: May refer to cell class (e.g. red blood cell) or cell line (e.g. u2os)
> * **cell_visible**: May refer to cell components that were stained (e.g. actin) or features of an organism that were stained for (e.g. pharynx)
> * **phenotype**: May refer to disease condition (e.g. leukemia), mechanism-of-action (MOA), cell cycle stage (e.g. telophase), etc.
> * **compound**: Name of compound treatment
> * **sirna**: Name of siRNA treatment
> * **gene**: Name of gene targeted (*" targeted" added to avoid overlapping labels with cell_visible*)

> * **microscopy**: Microscopy modality (e.g. fluorescence)
> * **crop**: True if image is a crop from an image. False, otherwise.
> * **scaling**: Length of crop side relative to size of original image (e.g. 0.5 corresponds to a 0.5 by 0.5 window ~ 1/4 the original image)
> * **x_min, x_max, y_min, y_max**: Slicing indices to create crop from original image

**NOTE**: In the case of multi-labels, labels are separated by a "|" (e.g. nucleus|actin|mitochondria).

**EXTRA NOTE**: All labels were converted to lowercase, which may make searching labels difficult, particularly with compound labels.

---
# Methods

## Data Cleaning

### Annotation
65 datasets were manually searched one by one, requiring dataset-specific annotation processing due to inconsistency and sometimes unreliability metadata. If metadata was available for all images, columns were selected and often transformed to create the standardized metadata for each image above. If not, metadata was curated based on available information about the dataset and assigned to all images in the dataset. Labels found in other datasets with different names were standardized and merged if found (e.g. neutrophil -> white blood cell, GPCR -> cell membrane). In the case that multiple labels exist for a category, multi-labels are separated by a "|" (e.g. nucleus|actin|mitochondria).

For fluorescent microscopy images, images are typically grayscale with around 1-7+ images depending on what is being stained for. Yet, these correspond to 1 row in the dataset. These pseudo (uncreated) images are given filenames that allow mapping to the original existing images to merge. These images with 1-7+ channels are merged as part of the data processing pipeline, if selected. 

In total, this produced **2.7 million rows** with each row corresponding to an image and a unique image index.

**RELEVANT CODE**: `clean_metadata.py`, `describe_dataset.py`

---

### Weak Label Assignment & Stratified Downsampling
Of the 2.7 million row table, each column from [organism, cell_type, cell_visible, phenotype, compound, gene, sirna] were searched for unique values and their counts, ignoring labels that overlap across columns. To create near to 1000 labels, potential labels with counts equal to or above a chosen threshold of 287 served as *potential labels*. Beginning from the least counts, potential labels were iterated through, keeping track of rows that were already assigned labels in a hash table via their unique index. Stratified sampling based on metadata curated is used to improve diversity of images selected for labels.

***Pseudo-Code***: for each potential label
1. Filter for images containing potential label in metadata AND not currently in hash table.
2. Skip if < 287 images for potential label. End iteration.
3. If less than 1000 images, skip to step 6.
4. If > 10,000 images, sample 10,000 rows by a preliminary stratified sampling on columns in [dataset, organism and cell type].
5. If > 1000 images, sample 1000 rows by stratified sampling on columns in [organism, cell_type, cell visible, sirna, compound, phenotype].
6. Save potential label and update hash table with used files.

**RELEVANT CODE**: `analyze_metadata.py`

---

### Image Data Cleaning & Standardization
In general, there is no one-size-fits-all when it comes to microscopy images since the types of images collected vary based on the study. And a lack of a golden standard for storing image data makes data cleaning a dataset-specific task. The following steps are done on selected images...

* **Standardize file format** to PNG from other formats (TIF, JPG, FLEX, BMP, etc.)
* Converting RGB images **to grayscale**.
* If merging fluorescently stained channels, **normalize** each channel using 0.1th and 99.9th percentile pixel intensity.
* **Merge** fluorescently stained channels to create grayscale images. 

**NOTE**: Brightfield microscopy images are separated from fluorescent microscopy images.

**NOTE**: Single channel images also exist (e.g. an image only stained for actin).

**EXTRA NOTE**: Normalization procedure follows preprocessing used in training [DeepLoc](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5408780/) and helps in brightening dim images.

> Merging Procedure:
![channel merging](https://user-images.githubusercontent.com/63123494/130717702-184e3b14-f4ad-4e27-b26b-a1d95e11c6e3.png)

**RELEVANT CODE**: `preprocessor.py`, `prepare_dataset.py`

---

### Upsampling Classes
![upsampling_procedure](https://user-images.githubusercontent.com/63123494/130723750-c7400d1a-31e1-4393-9ac3-e6f7117a58d7.png)

To increase image diversity and class size, we take 1-4 crops per image of different scale, preferably. Since microscopy images are not center-focused, we split the image into 4 non-overlapping windows. For each window, we randomly choose the crop size based on a scaling factor from [1/2, 1/4, 1/8, 1/16], making sure that crop sizes are above 70x70 pixels. Then we take the random crop from anywhere in the window. Crops are filtered for artifacts (completely white/black, too dark). Each crop is normalized once again with respect to the 0.1th and 99.9th percentile pixel intensity.

We extract ImageNet features and use UMAPs (a dimensionality reduction method) to visualize the effects of our chosen upsampling method on 20 randomly chosen classes. After upsampling, it becomes more difficult for ImageNet features to tell apart images in each class.

![upsampling_effects](https://user-images.githubusercontent.com/63123494/130719806-e36fe929-f4b0-49de-b19d-3a5203e71851.png)

**RELEVANT CODE**: `prepare_dataset.py`, `feature_extraction.py`, `visualize_classes.py`

---

## Model Training
Implemented in Tensorflow Keras, **EfficientNetB0** is the chosen convolutional neural network architecture to train on CytoImageNet. Its relatively small number of parameters allows for faster training times, and it favorably limits the amount of information that can be learnt.

**RELEVANT CODE**: `model_pretraining.py`

## Evaluation
We validate the performance of our trained features on the **BBBC021 evaluation protocol** from the Broad Institute. The general procedure is as follows:
1. Extract image features from ~2000 images (*each 'image' is made of 3 grayscale fluorescent microscopy images*).
2. Aggregate mean feature vector on treatment (compound - at specific concentration). Resulting in 103 feature vectors corresponding to 103 treatments.
3. Using **1-nearest neighbors**, classify mechanism-of-action (MOA) label, excluding neighbors with same compound treatments.
4. Report overall not-same-compound (NSC) accuracy.

**RELEVANT CODE**: `model_evaluation.py`
