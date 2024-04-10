# [RCDS](https://www.imperial.ac.uk/students/academic-support/graduate-school/professional-development/doctoral-students/research-computing-data-science/courses/), Deep Learning with Python

In this course, we will go through the basic concepts of Deep Learning, taking a hands-on approach. You will thus learn how to develop and apply convolutional neural networks (CNN) in Python using [PyTorch](https://pytorch.org/). For this purpose, we will delve into the basics of PyTorch and concepts of computer vision using [torchvision](https://pytorch.org/vision/stable/index.html) (lecture 1). We will train our own CNNs from scratch and use transfer learning to build on existing neural networks (lecture 2). Finally, you will learn how to optimise and evaluate your network architecture and its performance (lecture 3). 

Previous knowledge of PyTorch is not required for the course nor are you expected to know anything about Deep Learning beforemand, as we will cover the basics. However, participants must be familiar with the basic concepts of Python programming. Furthermore, since we will build on the terminology introduced in the RCDS course [Introduction to Machine Learning](https://github.com/ImperialCollegeLondon/RCDS-intro-to-machine-learning), it is a prerequisite for the present workshop.

The slides from the lectures are provided in the folder SLIDES.

The folder NOTEBOOKS contains Jupyter-notebooks with the examples and exercises used in the course.

In the folder QUIZ, you will find a notebook that provides an interactive multiple-choice quiz on the covered material. 

## Learning outcomes

After attending this workshop, you will be better able to:

- understand the basic terminology and concepts of deep learning methods 

- understand and explain the applications and limitations of CNN 

- select, apply and develop deep learning techniques in PyTorch.  

- evaluate the performance of CNN 

## Getting Started

If you want to run the notebooks on your own laptop, you will need to [install PyTorch](https://pytorch.org/). In the course, we will run the code on Google Colab, i.e., no local installations will be required. <span style="background-color:lightblue;">See the notebook "0.1 Installation" in NOTEBOOKS</span>.

## Suggested external resources

For a general introduction to machine learning beyond Deep Learning, please see 

- the RCDS course [Introduction to Machine Learning](https://github.com/ImperialCollegeLondon/RCDS-intro-to-machine-learning).

- the RCDS course [Machine Learning with Python](https://github.com/ImperialCollegeLondon/RCDS-machine-learning-with-python).

For more information on PyTorch, see also [Learn PyTorch for Deep Learning: Zero to Mastery](https://www.learnpytorch.io/).

Due to the time constraints, the course delves into CNNs only. The course does hence not cover Physics-informed neural networks (PINN), Transformers, Graph neural networks (GNN), Generative Adversarial Networks (GAN), normalising flows, or autoencoders. For **post-course reading** on any of these topics, we recommend

- The book [Dive into Deep Learning](https://d2l.ai/index.html) by Aston Zhang et al. (includes coding tutorials)

- The book [Deep Learning, Foundations and Concepts](https://link.springer.com/book/10.1007/978-3-031-45468-4) by Bishop and Bishop

- The book Deep Learning by Ian Goodfellow, Yoshua Bengio, Aaron Courville and Francis Bach

For a visual representation of what happens in each layer of a CNN, have a look at [tensorspace.org](tensorspace.org). You can create such a visual representation yourself (see notebook 2.1).

For animations of what happens inside a neural network, have a look at [https://animatedai.github.io](https://animatedai.github.io).

In this course, you will find code that takes you through each step. However, you can get Python packages, such as [Supergradient](https://pypi.org/project/super-gradients/2.5.0/), that allow you to train your model with a single line of code.

## Lectures

If you are a student or member of staff at Imperial College London, you can access the lectures recordings on Panopto

- [Lecture 1, PyTorch and Deep Learning](https://imperial.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=a7692ac6-e98c-4d79-ac18-b14d00a8ca1b)

- [Lecture 2, Convolutional Neural Networks](https://imperial.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=6397640e-1cb7-4e77-8b38-b14d00ab3766)

- [Lecture 3, Training, testing and optimisation](https://imperial.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=180f480f-baf9-45db-b321-b14d00c20c6c)

## Google Colab

You can access the notebooks directly on Google Colab via the links below to create your own copy of the notebooks. Alternatively, you can find the code in the corresponding folders as listed above. Note that the notebooks are tailored for Google Colab, i.e., if you decide to run the code on your own computer, you might need to adapt a few lines of code (e.g., to load the files in the quiz).

- Notebook 0.1: <a href="https://colab.research.google.com/drive/1Cu213KYxYPqALhrZ5reDtWtr4bbwMDoP?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Notebook 1.1: <a href="https://colab.research.google.com/drive/1GgEcCmukBWUbGatYA1kyU_bo4UmxZVLI?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Notebook 1.2: <a href="https://colab.research.google.com/drive/1JvNk2XSm5NY0F9kbtBQxA04n90hJA5mN?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Notebook 1.3: <a href="https://colab.research.google.com/drive/12gRIGNnjbCA-1Hkt97fjB7NU9CnvYt8V?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Notebook 2.1: <a href="https://colab.research.google.com/drive/1ddF3Rkcag9ywO2XovA7ONNHDKRgid5bv?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Notebook 2.2: <a href="https://colab.research.google.com/drive/1jd3HeWWEb78zTFBIQYPRjTECjkfJC1yV?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Notebook 2.3: <a href="https://colab.research.google.com/drive/1t1a_yLCAcqtIl9uIqkE01A8VwWKgbjSc?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Notebook 3.1: <a href="https://colab.research.google.com/drive/1TzlwecBPvyCVqTBf4z9fpgo5afJIi0_4?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Notebook 3.2: <a href="https://colab.research.google.com/drive/1sdjecxIHi-7x-KxIydD0TR9gqOdmGmDe?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Quiz notebook: <a href="https://colab.research.google.com/drive/1qXrvMNi6Z9rkZ6RHllbOwmfQ_4knzGFX?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
