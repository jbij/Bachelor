This repository contains the code and final report of our bachelor thesis at the IT University of Copenhagen. The thesis was supervised by Luca Aiello.


This repository contains the source code and documentation for our bachelor’s thesis project, "Enabling user-driven exploration: Structural and Semantic Analysis of
Interaction Networks on TikTok. 

The project entailed the collection and transformation of a dataset of videos relating to climate change topics. For the purposes of analyzing user interactions, we scraped additional data through accessing the API domain
https://open.tiktokapis.com using client credentials, which were requested for the creation of the
original data.

This project aims to develop transparent, user-centered tools that enable semantic exploration of content on TikTok, with a focus on the platform's climate change discourse. By leveraging TikTok's newly released API, we investigate alternative ways to model relationships between hashtags—based on semantic similarity, co-occurrence, and user interaction—beyond the opaque mechanisms of algorithmic recommendation systems.
Through this work, we seek to contribute to both methodological innovation in content recommendation and critical discourse around algorithmic transparency in digital platforms

SETUP
python version

The backbone of this project is written in Python. Make sure that you have the correct Python version (3.11) by running python --version.

# Core scientific stack
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.2
scikit-learn>=1.1.0
scipy>=1.9.0

# Network analysis and graph tools
networkx>=2.8
python-igraph>=0.10
leidenalg>=0.9.0
python-louvain>=0.16  # community module
node2vec>=0.4.6

# Statistical and power-law tools
powerlaw>=1.5
joypy>=0.2.6
tqdm>=4.64.0

# Sentence embeddings
sentence-transformers>=2.2.0

# Visualization tools
matplotlib-venn>=0.11.7

# Optional graph visualization (may require system-level Graphviz install)
pygraphviz>=1.9

# Backboning
We use a backboning script to extract the significant structure from co-occurrence networks. The script is based on the implementation by Michele Coscia, available at [https://www.michelecoscia.com/?page_id=287](https://www.michelecoscia.com/?page_id=287). Credit to the original author for the method and code.
