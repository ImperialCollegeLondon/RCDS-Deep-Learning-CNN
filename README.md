# [RCDS](https://www.imperial.ac.uk/students/academic-support/graduate-school/professional-development/doctoral-students/research-computing-data-science/courses/), Introduction to Deep Learning and convolutional neural networks in Python

In this workshop, we will go through the basic concepts of Deep Learning, taking a hands-on approach. You will thus learn how to develop and apply convolutional neural networks (CNN) in Python using [PyTorch](https://pytorch.org/). For this purpose, we will delve into the basics of PyTorch and concepts of computer vision using [torchvision](https://pytorch.org/vision/stable/index.html) (lecture 1). We will train our own CNN from scratch and use transfer learning to build on existing neural netoworks (lecture 2). Finally, you will learn how to optimise and evaluate your network architecture and its perfromance (lecture 3). 

Previous knowledge of PyTorch is not required for the course. However, participants must be familiar with the basic concepts of Python programming. Furthermore, since we will build on the terminology introduced in the RCDS course [Introduction to Machine Learning](https://github.com/ImperialCollegeLondon/RCDS-intro-to-machine-learning), it is a prerequisite for the present workshop.

The slides from the lectures are provided in the folder SLIDES.

The folder NOTEBOOKS contains Jupyter-notebooks with the examples and exercises used in the course.

In the folder QUIZ, you will find a notebook providing an interactive multiple-choice quiz on the covered material. 

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

- RCDS course [Introduction to Machine Learning](https://github.com/ImperialCollegeLondon/RCDS-intro-to-machine-learning).

- please see the RCDS course [Machine Learning with Python](https://github.com/ImperialCollegeLondon/RCDS-machine-learning-with-python).

Due to the time constraints, the course delves into CNNs only. The course does hence not cover Physics-informed neural networks (PINN), Transformers, Graph neural networks (GNN), Generative Adversarial Networks (GAN), normalising flows, or autoencoders. For **post-course reading** on any of these topics, we recommend

- The book [Dive into Deep Learning](https://d2l.ai/index.html) by Aston Zhang et al. (includes coding tutorials)

- The book [Deep Learning, Foundations and Concepts](https://link.springer.com/book/10.1007/978-3-031-45468-4) by Bishop and Bishop

- The book Deep Learning by Ian Goodfellow, Yoshua Bengio, Aaron Courville and Francis Bach

For a visual representation of what happens in each layer of a CNN, have a look at [tensorspace.org](tensorspace.org). You can create such a visiual representation yourself (see notebook 2.1).

For animations of what happens inside a neural network, have a look at [https://animatedai.github.io](https://animatedai.github.io)
