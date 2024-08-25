# The Representation Landscape of Few-Shot Learning and Fine-Tuning in Large Transformer Models

This repository contains the source code for the paper: "The Representation Landscape of Few-Shot Learning and Fine-Tuning in Large Transformer Models."

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This project provides the implementation details and source code for the research conducted on the representation landscape of few-shot learning and fine-tuning in large transformer models. The aim is to explore how these models adapt to new tasks with minimal data and the impact of fine-tuning on their performance and representations.

## Installation

To set up the project, you will need to install [Poetry](https://python-poetry.org/), which is a tool for dependency management and packaging in Python. Follow these steps to install the virtual environment and dependencies:

1. **Install Poetry** (if you don't have it installed yet):
   ```sh
   curl -sSL https://install.python-poetry.org | python3 -
2. **Clone the repository**
   ```sh
   git clone git@github.com:diegodoimo/representation_landscape_fs_ft.git
   cd representation_landscape_fs_ft
4. **Install the dependencies and create the virtual environment**:
   ```sh
   poetry install
6. **Activate the virtual environment**:
   ```sh
   poetry shell

## Usage

1. **Run a script**:
   ```sh
    poetry run python your_script.py
3. **Open a Jupyter notebook**
   ```sh
    poetry run jupyter notebook
