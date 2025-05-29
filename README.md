# BIMCompNet: An IFC-Derived Multimodal Dataset for Geometric Deep Learning in BIM

## 📦 Resources Overview

This GitHub repository provides **all code and scripts** related to the **BIMCompNet** project, including:

- ✅ **Dataset construction pipeline** (model-level and component-level)
- ✅ **Data balancing strategies**
- ✅ **Baseline model training and evaluation**

### 📁 Source Code Repository (current page)  
You are here: full codebase for dataset processing and deep learning experiments.

### 🌐 Full Dataset Download  
Available at the official BIMCompNet dataset site, with per-building-type subsets and multiple sampling scales:  
👉 [https://bimcompnet-606lab.xaut.edu.cn/](https://bimcompnet-606lab.xaut.edu.cn/)

### 🤖 100-Sample Subset + Pretrained Models  
Designed for fast benchmarking and reproducible evaluation:  
👉 [https://huggingface.co/datasets/YaMiSo1995/BIMCompNet_100-samplesubset](https://huggingface.co/datasets/YaMiSo1995/BIMCompNet_100-samplesubset)

---

📊 Benchmark Results on BIMCompNet
The following summarizes the classification performance (F1-score) of various baseline models across different sampling scales. 

| Model       | View Sampling    | 100 / class | 500 / class | 1000 / class | 5000 / class |
|-------------|------------------|-------------|-------------|--------------|--------------|
| **MVCNN**   | IfcNet           | 75.76       | 87.00       | 88.38        | 89.09        |
|             | ArchShapeNet     | 76.26       | 87.34       | 88.73        | 89.12        |
|             | **Edges**        | **76.34**   | **87.41**   | **89.32**    | **89.39**    |
|             | Faces            | 74.91       | 86.81       | 88.92        | 89.27        |
|             | Vertices         | 74.21       | 86.29       | 88.38        | 88.86        |
| **DGCNN**   | /                | 65.92       | 79.44       | 84.23        | 86.94        |
| **MeshNet** | /                | 72.10       | 84.86       | 86.85        | 88.57        |
| **VoxNet**  | /                | 62.85       | 79.09       | 84.06        | 85.39        |
| **RGCN**    | /                | **74.80**   | **86.43**   | **87.94**    | **88.84**    |

📝 Metric shown: macro F1-score (%)
📥 You can view the full experiment table in the paper.

---

## 🔗 Training RGCN with Graph Input

Before training the RGCN model using the Graph input, you must copy the REVERSE_IFCSchemaGraph.bin file from the data directory into the graph data directory of each instance to be used. This file contains the reverse IFC schema graph and is required for the model to correctly interpret heterogeneous structures during training.

We provide a utility script utils/file_copy_delete.py to automate this process. Run the script before training to broadcast the file to all instance-level graph directories.

If this step is skipped, the model may fail to load the necessary schema structure, resulting in errors or invalid training behavior.

---

## 📚 Citation

If you use BIMCompNet in your research, please cite our work using the formats provided on the [official dataset website](https://bimcompnet-606lab.xaut.edu.cn/#citation).

Two BibTeX entries are available:
- A general dataset citation (`@misc`)
- A conference paper under review (`@unpublished`)

📌 Citation details are shown at the bottom of the dataset webpage.

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.
