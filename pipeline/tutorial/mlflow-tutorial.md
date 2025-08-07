Getting Started with MLflow
If you're new to MLflow or seeking a refresher on its core functionalities, the quickstart tutorials here are the perfect starting point. They will guide you step-by-step through fundamental concepts, focusing purely on a task that will maximize your understanding of how to use MLflow to solve a particular task.

tip
If you'd like to try a free trial of a fully-managed MLflow experience on Databricks, you can quickly sign up and start using MLflow for your GenAI and ML project needs without having to configure, setup, and run your own tracking server. You can learn more about the

Databricks Free Trial here. This trial offers full access to a personal Databricks account that includes MLflow and other tightly integrated AI services and features.

Guidance on Running Tutorials
If you have never interfaced with the MLflow Tracking Server, we highly encourage you to head on over to quickly read the guide below. It will help you get started as quickly as possible with tutorial content throughout the documentation.

How to Run Tutorials
Learn about your options for running a MLflow Tracking Server for executing any of the guides and tutorials in the MLflow documentation


Getting Started Guides
MLflow Tracking
MLflow Tracking is one of the primary service components of MLflow. In these guides, you will gain an understanding of what MLflow Tracking can do to enhance your MLOps related activities while building ML models.

The basics of MLflow tracking

In these introductory guides to MLflow Tracking, you will learn how to leverage MLflow to:

Log training statistics (loss, accuracy, etc.) and hyperparameters for a model
Log (save) a model for later retrieval
Register a model using the MLflow Model Registry to enable deployment
Load the model and use it for inference
In the process of learning these key concepts, you will be exposed to the MLflow Tracking APIs, the MLflow Tracking UI, and learn how to add metadata associated with a model training event to an MLflow run.

MLFlow Tracking Quickstart Guide
Learn the basics of MLflow Tracking in a fast-paced guide with a focus on seeing your first model in the MLflow UI

In-depth Tutorial for MLflow Tracking
Learn the nuances of interfacing with the MLflow Tracking Server in an in-depth tutorial

Autologging Basics
A great way to get started with MLflow is to use the autologging feature. Autologging automatically logs your model, metrics, examples, signature, and parameters with only a single line of code for many of the most popular ML libraries in the Python ecosystem.

The basics of MLflow tracking

In this brief tutorial, you'll learn how to leverage MLflow's autologging feature to simplify your model logging activities.

MLflow Autologging Quickstart
Get started with logging to MLflow with the high-level autologging API in a fast-paced guide


Run Comparison Basics
This quickstart tutorial focuses on the MLflow UI's run comparison feature and provides a step-by-step walkthrough of registering the best model found from a hyperparameter tuning execution sweep. After locally serving the registered model, a brief example of preparing a model for remote deployment by containerizing the model using Docker is covered.

The basics of MLflow run comparison

MLflow Run Comparison Quickstart
Get started with using the MLflow UI to compare runs and register a model for deployment


Tracking Server Quickstart
This quickstart tutorial walks through different types of MLflow Tracking Servers and how to use them to log your MLflow experiments.

5 Minute Tracking Server Overview
Learn how to log MLflow experiments with different tracking servers


Model Registry Quickstart
This quickstart tutorial walks through registering a model in the MLflow model registry and how to retrieve registered models.

5 Minute Model Registry Overview
Learn how to log MLflow models to the model registry

Further Learning - What's Next?
Now that you have the essentials under your belt, below are some recommended collections of tutorial and guide content that will help to broaden your understanding of MLflow and its APIs.

Tracking - Learn more about the MLflow tracking APIs by reading the tracking guide.
MLflow Deployment - Follow the comprehensive guide on model deployment to learn how to deploy your MLflow models to a variety of deployment targets.
Model Registry - Learn about the MLflow Model Registry and how it can help you manage the lifecycle of your ML models.
Deep Learning Library Integrations - From PyTorch to TensorFlow and more, learn about the integrated deep learning capabilities in MLflow by reading the deep learning guide.
Traditional ML - Learn about the traditional ML capabilities in MLflow and how they can help you manage your traditional ML workflows.


How to Run Tutorials
This brief guide will walk you through some options that you have to run these tutorials and have a Tracking Server that is available to log the results to (as well as offering options for the MLflow UI).

1. Download the Notebook
You can download the tutorial notebook by clicking on the "Download the Notebook" button at the top of each tutorial page.

2. Install MLflow
Install MLflow from PyPI by running the following command:

pip install mlflow

tip
💡 We recommend creating a new virtual environment with tool like venv, uv, Poetry, to isolate your ML experiment environment from other projects.

3. Set Up MLflow Tracking Server (Optional)
Direct Access Mode (no tracking server)
You can start a tutorial and log models, experiments without a tracking server set up. With this mode, your experiment data and artifacts are saved directly under your current directory.

While this mode is the easiest way to get started, it is not recommended for general use:

The experiment data is only visible on the UI when you run mlflow ui command from the same directory.
The experiment data cannot be accessible from other team members.
You may lose your experiment data if you accidentally delete the directory.
Running with Tracking Server (recommended)
MLflow Tracking Server is a centralized HTTP server that allows you to access your experiments artifacts regardless of where you run your code. To use the Tracking Server, you can either run it locally or use a managed service. Click on the following links to learn more about each option:

MLflow can be used in a variety of environments, including your local environment, on-premises clusters, cloud platforms, and managed services. Being an open-source platform, MLflow is vendor-neutral; no matter where you are doing machine learning, you have access to the MLflow's core capabilities sets such as tracking, evaluation, observability, and more.

Databricks Logo
Amazon SageMaker Logo
Azure Machine Learning Logo
Nebius Logo
Kubernetes Logo

tip
💡 If you are not a Databricks user but want to try out its managed MLflow experience for free, we recommend using the

Databricks Free Trial . It offers a personal Databricks account with service credits that are valid for a limited time period, letting you get started with MLflow and other ML/AI tools in minutes.

4. Connecting Notebook with Tracking Server
note
If you are on Databricks, you can skip this section because a notebook is already connected to the tracking server and an MLflow experiment.

For other managed service (e.g., SageMaker, Nebius), please refer to the documentation provided by the service provider.

Once the tracking server is up, connect your notebook to the server by calling mlflow.set_tracking_uri("http://<your-mlflow-server>:<port>"). For example, if you are running the tracking server locally on port 5000, you would call:

mlflow.set_tracking_uri("http://localhost:5000")

This will ensure that all MLflow runs, experiments, traces, and artifacts are logged to the specified tracking server.

5. Ready to Run!
Now that you have set up your MLflow environment, you are ready to run the tutorials. Go back to the tutorial and enjoy the journey of learning MLflow!🚢

Previous
Getting Started with MLflow
Next
Your First MLflow Model: Complete Tutorial

Your First MLflow Model: Complete Tutorial
Master the fundamentals of MLflow by building your first end-to-end machine learning workflow. This hands-on tutorial takes you from setup to deployment, covering all the essential MLflow concepts you need to succeed.

What You'll Build
By the end of this tutorial, you'll have created a complete ML pipeline that:

🎯 Predicts apple quality using a synthetic dataset you'll generate
📊 Tracks experiments with parameters, metrics, and model artifacts
🔍 Compares model performance using the MLflow UI
📦 Registers your best model for production use
🚀 Deploys a working API for real-time predictions
Perfect for Beginners
🎓 No prior MLflow experience required. We'll guide you through every concept with clear explanations and practical examples.

⏱️ Complete the full tutorial at your own pace in 30-45 minutes, with each step building naturally on the previous one.

Learning Path
This tutorial is designed as a progressive learning experience:

Phase 1: Setup & Foundations (10 minutes)
🖥️ Start Your MLflow Tracking Server - Get your local environment running
🔌 Master the MLflow Client API - Learn the programmatic interface
📁 Understand MLflow Experiments - Organize your ML work
Phase 2: Data & Experimentation (15 minutes)
🔍 Search and Filter Experiments - Navigate your work efficiently
🍎 Generate Your Apple Dataset - Create realistic training data
📈 Log Your First ML Runs - Track parameters, metrics, and models
What Makes This Tutorial Special
Real-World Focused
Instead of toy examples, you'll work with a realistic apple quality prediction problem that demonstrates practical ML workflows.

Hands-On Learning
Every concept is immediately applied through code examples that you can run and modify.

Complete Workflow
Experience the full ML lifecycle from data creation to model deployment, not just isolated features.

Visual Learning
Extensive use of the MLflow UI helps you understand how tracking data appears in practice.

Prerequisites
Python 3.8+ installed on your system
Basic Python knowledge (variables, functions, loops)
10 minutes for initial setup
No machine learning expertise required - we'll explain the ML concepts as we go!

Two Ways to Follow Along
Interactive Web Tutorial (Recommended)
Follow the step-by-step guide in your browser with detailed explanations and screenshots. Perfect for understanding concepts deeply.

▶️ Start the Interactive Tutorial

Jupyter Notebook
Download and run the complete tutorial locally. Great for experimentation and customization.

Key Concepts You'll Master
🖥️ MLflow Tracking Server Set up and connect to the central hub that stores all your ML experiments and artifacts.

🔬 Experiments & Runs Organize your ML work into logical groups and track individual training sessions with complete reproducibility.

📊 Metrics & Parameters Log training performance, hyperparameters, and model configuration for easy comparison and optimization.

🤖 Model Artifacts Save trained models with proper versioning and metadata for consistent deployment and sharing.

🏷️ Tags & Organization Use tags and descriptions to keep your experiments organized and searchable as your projects grow.

🔍 Search & Discovery Find and compare experiments efficiently using MLflow's powerful search and filtering capabilities.

What Happens Next
After completing this tutorial, you'll be ready to:

Apply MLflow to your own projects with confidence in the core concepts
Explore advanced features like hyperparameter tuning and A/B testing
Scale to team workflows with shared tracking servers and model registries
Deploy production models using MLflow's serving capabilities
Ready to Begin?
Choose your preferred learning style and dive in! The tutorial is designed to be completed in one session, but you can also bookmark your progress and return anytime.

Get Started Now
Interactive Tutorial: 🚀 Start Step 1 - Tracking Server

Notebook Version: Use the download button above to get the complete Jupyter notebook

Questions or feedback? This tutorial is continuously improved based on user input. Let us know how we can make your learning experience even better!