# CytoImageNet
A pretraining dataset for bioimage transfer learning. 

Motivation
---
In past few decades, the increase in speed of data collection has led to the dawn of so-called '*big data*'. In the field of molecular biology, 
this was seen in 'high throughput sequencing', where DNA and protein assays exceeded the capability of scientists to analyze large amount of datas.
The need to develop computational methods that can match the rate at which data is collected sparked the era of 'bioinformatics'. In more recent years,
our ability to capture and store biological images has grown tremendously. 

There is now a need for **automated methods to help analyze biological images**.
Here, we take inspiration from the success of ImageNet to curate CytoImageNet; a large-scale dataset of weakly labeled microscopy images.

## About the data
**890,217** total images. **894 classes** (~1000 per class).

Microscopy images belong to **40 openly available datasets** from the following databases: Recursion, Image Data Resource, Broad Bioimage Benchmark Collection,
Kaggle and the Cell Image Library.

The classes are soft/weak labels, and it's possible for overlap. Labels were assigned based on image metadata provided in the originating datasets. Chosen label could 
correspond to any of [organism, cell_type, cell_visible, phenotype, compound, gene, sirna].

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

**EXTRA NOTE**: All labels were converted to lowercase, potentially leading to obscure labels, particularly with compound.

---
# Methods

## Data Cleaning

### Annotation
### Weak Label Assignment
### Image Data Cleaning & Standardization
### Upsampling
## Model Training
