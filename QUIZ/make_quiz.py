import pandas as pd

# PyTorch quiz data
pytorch_quiz_data = [
        {
    'question': 'What is the PyTorch command used to define CrossEntropy Loss?',
    'options': 'torch.loss.CrossEntropy(),torch.nn.CrossEntropy(),torch.CrossEntropyLoss(),torch.nn.CrossEntropyLoss()',
    'correct_answer': 'torch.nn.CrossEntropyLoss()'
},

        {'question': 'What PyTorch command is commonly used to save a trained model?',
    'options': 'torch.save_model(),torch.export_model(),torch.store_model(),torch.save()',
    'correct_answer': 'torch.save()'},
    {
        'question': 'What is PyTorch primarily used for?',
        'options': 'Computer vision,Text processing,Speech recognition,Reinforcement learning',
        'correct_answer': 'Computer vision'
    },
    {    'question': 'What does the `torch.Tensor` class represent in PyTorch?',
        'options': 'A deep learning model,A neural network layer,A multi-dimensional array,A Python function',
        'correct_answer': 'A multi-dimensional array'
    },
    {
        'question': 'Which module is commonly used for loading datasets in PyTorch?',
        'options': 'torch.utils.data,torch.nn,torch.optim,torch.autograd',
        'correct_answer': 'torch.utils.data'
    },
    {
        'question': 'What is the purpose of the `torch.nn.Module` class in PyTorch?',
        'options': 'To define a neural network model,To handle automatic differentiation,To optimize the model parameters,To preprocess data',
        'correct_answer': 'To define a neural network model'
    },
    {
        'question': 'Which activation function is commonly used in the hidden layers of a neural network?',
        'options': 'ReLU,Sigmoid,Tanh,Softmax',
        'correct_answer': 'ReLU'
    },
    {
        'question': 'What is the purpose of the `torch.optim` module in PyTorch?',
        'options': 'To define loss functions,To perform automatic differentiation,To optimize the model parameters,To preprocess data',
        'correct_answer': 'To optimize the model parameters'
    },
    {
        'question': 'What does the DataLoader class do in PyTorch?',
        'options': 'Loads and preprocesses data for training,Defines the model architecture,Performs automatic differentiation,Optimizes the model parameters',
        'correct_answer': 'Loads and preprocesses data for training'
    },
    {
        'question': 'Which PyTorch function is used to transfer a tensor to a GPU?',
        'options': 'tensor.to_gpu(),tensor.cuda(),torch.to_gpu(),torch.cuda()',
        'correct_answer': 'tensor.cuda()'
    },
    {
        'question': 'What is the purpose of the CrossEntropyLoss in PyTorch?',
        'options': 'Classification loss for multiple classes,Regression loss,Binary classification loss,Mean squared error loss',
        'correct_answer': 'Classification loss for multiple classes'
    },
    {
        'question': 'How can you check the size of a tensor in PyTorch?',
        'options': 'tensor.size(),tensor.shape(),tensor.length(),tensor.dim()',
        'correct_answer': 'tensor.shape()'
    },
    {
        'question': 'What is the purpose of the flatten operation in a neural network?',
        'options': 'To increase the number of parameters,To decrease the number of parameters,To transform a multi-dimensional tensor into a one-dimensional tensor,To add non-linearity to the model',
        'correct_answer': 'To transform a multi-dimensional tensor into a one-dimensional tensor'
    },
    {
        'question': 'In PyTorch, what is an epoch in training a neural network?',
        'options': 'One complete pass through the entire training dataset,One iteration of the optimization algorithm,One forward and backward pass,One update of the model parameters',
        'correct_answer': 'One complete pass through the entire training dataset'
    },
    {
        'question': 'Which layer is commonly used for downsampling in a convolutional neural network (CNN)?',
        'options': 'MaxPooling2D,Flatten,Dropout,ReLU',
        'correct_answer': 'MaxPooling2D'
    },
    {
        'question': 'What is the purpose of the `torchvision` library in PyTorch?',
        'options': 'Provides pre-trained models and datasets for computer vision,Optimizes model parameters,Defines loss functions,Handles automatic differentiation',
        'correct_answer': 'Provides pre-trained models and datasets for computer vision'
    },
    {
        'question': 'In PyTorch, how do you define a custom dataset class for loading images?',
        'options': 'Subclassing `torch.utils.data.Dataset` and implementing `__len__` and `__getitem__`,Using the `torch.nn.Module` class,Defining a Python function,Using the `torchvision` library',
        'correct_answer': 'Subclassing `torch.utils.data.Dataset` and implementing `__len__` and `__getitem__`'
    },
    {
        'question': 'What is the purpose of the Batch Normalization layer in a neural network?',
        'options': 'Improves convergence during training,Reduces the number of parameters,Adds non-linearity to the model,Increases the learning rate',
        'correct_answer': 'Improves convergence during training'
    },
    {
        'question': 'How can you initialize the weights of a neural network in PyTorch?',
        'options': 'torch.init_weights(),torch.initialize_weights(),torch.nn.init(),torch.weights_init()',
        'correct_answer': 'torch.nn.init()'
    },
        {
        'question': 'Which function is used to perform a forward pass in PyTorch?',
        'options': 'model.forward(),model.backward(),model.predict(),model.training()',
        'correct_answer': 'model.forward()'
    },
    {
        'question': 'What is the purpose of the Rectified Linear Unit (ReLU) activation function?',
        'options': 'Introduces non-linearity,Ensures smooth gradients,Reduces overfitting,Normalizes the input',
        'correct_answer': 'Introduces non-linearity'
    },
    {
        'question': 'How do you define a convolutional layer in PyTorch?',
        'options': 'torch.FullyConnectedLayer(),torch.Conv2d(),torch.MaxPooling2d(),torch.ActivationLayer()',
        'correct_answer': 'torch.Conv2d()'
    },
    {
        'question': 'What is the purpose of the `torchvision.transforms` module?',
        'options': 'Applies transformations to images,Defines loss functions,Optimizes model parameters,Handles automatic differentiation',
        'correct_answer': 'Applies transformations to images'
    },
    {
        'question': 'How do you perform transfer learning in PyTorch using a pre-trained model?',
        'options': 'Load the pre-trained model and replace the final fully-connected layer,Load the pre-trained model and replace all layers,Train the model from scratch,Use the pre-trained model without any modifications',
        'correct_answer': 'Load the pre-trained model and replace the final fully-connected layer'
    },
    {
        'question': 'What is the purpose of the `torch.nn.CrossEntropyLoss` in PyTorch?',
        'options': 'Combines softmax activation and negative log-likelihood loss,Calculates mean squared error loss,Applies binary cross-entropy loss,Defines a custom loss function',
        'correct_answer': 'Combines softmax activation and negative log-likelihood loss'
    },
    {
        'question': 'In PyTorch, how do you calculate the number of parameters in a neural network model?',
        'options': 'Sum the number of parameters in each layer,Count the number of layers,Use the `torch.nn.parameter_count()` function,It is not possible to calculate the number of parameters',
        'correct_answer': 'Sum the number of parameters in each layer'
    },
    {
        'question': 'How do you visualize filters learned by a convolutional layer in PyTorch?',
        'options': 'Extract and display the learned filters using matplotlib,Use the `torch.visualization` module,Print the filters using the `print` function,There is no way to visualize learned filters',
        'correct_answer': 'Extract and display the learned filters using matplotlib'
    },
    {
        'question': 'What is the purpose of the ReLU activation function in a neural network?',
        'options': 'Introduces non-linearity,Ensures smooth gradients,Reduces overfitting,Normalizes the input',
        'correct_answer': 'Introduces non-linearity'
    },
    {
        'question': 'How can you check if a GPU is available in PyTorch?',
        'options': 'torch.has_gpu(),torch.cuda.is_available(),torch.detect_gpu(),torch.gpu_count()',
        'correct_answer': 'torch.cuda.is_available()'
    },
    {
        'question': 'What is the purpose of the `torch.nn.Dropout` layer?',
        'options': 'Reduces overfitting by randomly setting a fraction of input units to zero during training,Increases the learning rate,Normalizes the input,Adds non-linearity to the model',
        'correct_answer': 'Reduces overfitting by randomly setting a fraction of input units to zero during training'
    },
    {
        'question': 'Which loss function is commonly used for binary classification in PyTorch?',
        'options': 'Binary Cross-Entropy Loss,Hinge Loss,Squared Error Loss,Categorical Cross-Entropy Loss',
        'correct_answer': 'Binary Cross-Entropy Loss'
    },
    {
        'question': 'What does the `torchvision.models` module provide in PyTorch?',
        'options': 'Pre-trained models,Activation functions,Optimizers,Data loaders',
        'correct_answer': 'Pre-trained models'
    },
    {
        'question': 'In PyTorch, what is the purpose of the Batch Normalization layer in a neural network?',
        'options': 'Improves convergence during training,Reduces the number of parameters,Adds non-linearity to the model,Increases the learning rate',
        'correct_answer': 'Improves convergence during training'
    },
    {
        'question': 'How can you initialize the weights of a neural network in PyTorch?',
        'options': 'torch.init_weights(),torch.initialize_weights(),torch.nn.init(),torch.weights_init()',
        'correct_answer': 'torch.nn.init()'
    },
    {
        'question': 'Which function is used to perform a forward pass in PyTorch?',
        'options': 'model.forward(),model.backward(),model.predict(),model.training()',
        'correct_answer': 'model.forward()'
    },
    {
        'question': 'What is the purpose of the Rectified Linear Unit (ReLU) activation function?',
        'options': 'Introduces non-linearity,Ensures smooth gradients,Reduces overfitting,Normalizes the input',
        'correct_answer': 'Introduces non-linearity'
    },  
        {
        'question': 'How do you define a convolutional layer in PyTorch?',
        'options': 'torch.FullyConnectedLayer(),torch.Conv2d(),torch.MaxPooling2d(),torch.ActivationLayer()',
        'correct_answer': 'torch.Conv2d()'
    },
    {
        'question': 'What is the purpose of the `torchvision.transforms` module?',
        'options': 'Applies transformations to images,Defines loss functions,Optimizes model parameters,Handles automatic differentiation',
        'correct_answer': 'Applies transformations to images'
    },
    {
        'question': 'Which optimizer is commonly used for training deep neural networks in PyTorch?',
        'options': 'Adam,Stochastic Gradient Descent (SGD),RMSprop,Adagrad',
        'correct_answer': 'Adam'
    },
    {
        'question': 'How do you perform transfer learning in PyTorch using a pre-trained model?',
        'options': 'Load the pre-trained model and replace the final fully-connected layer,Load the pre-trained model and replace all layers,Train the model from scratch,Use the pre-trained model without any modifications',
        'correct_answer': 'Load the pre-trained model and replace the final fully-connected layer'
    },
    {
        'question': 'What is the purpose of the `torch.nn.CrossEntropyLoss` in PyTorch?',
        'options': 'Combines softmax activation and negative log-likelihood loss,Calculates mean squared error loss,Applies binary cross-entropy loss,Defines a custom loss function',
        'correct_answer': 'Combines softmax activation and negative log-likelihood loss'
    },
    {
        'question': 'In PyTorch, how do you calculate the number of parameters in a neural network model?',
        'options': 'Sum the number of parameters in each layer,Count the number of layers,Use the `torch.nn.parameter_count()` function,It is not possible to calculate the number of parameters',
        'correct_answer': 'Sum the number of parameters in each layer'
    },
    {
        'question': 'How do you visualize filters learned by a convolutional layer in PyTorch?',
        'options': 'Extract and display the learned filters using matplotlib,Use the `torch.visualization` module,Print the filters using the `print` function,There is no way to visualize learned filters',
        'correct_answer': 'Extract and display the learned filters using matplotlib'
    },
    {
        'question': 'What is the purpose of the ReLU activation function in a neural network?',
        'options': 'Introduces non-linearity,Ensures smooth gradients,Reduces overfitting,Normalizes the input',
        'correct_answer': 'Introduces non-linearity'
    },
    {
        'question': 'How can you check if a GPU is available in PyTorch?',
        'options': 'torch.has_gpu(),torch.cuda.is_available(),torch.detect_gpu(),torch.gpu_count()',
        'correct_answer': 'torch.cuda.is_available()'
    },
    {
        'question': 'What is the purpose of the `torch.nn.Dropout` layer?',
        'options': 'Reduces overfitting by randomly setting a fraction of input units to zero during training,Increases the learning rate,Normalizes the input,Adds non-linearity to the model',
        'correct_answer': 'Reduces overfitting by randomly setting a fraction of input units to zero during training'
    },
    {
        'question': 'Which loss function is commonly used for binary classification in PyTorch?',
        'options': 'Binary Cross-Entropy Loss,Hinge Loss,Squared Error Loss,Categorical Cross-Entropy Loss',
        'correct_answer': 'Binary Cross-Entropy Loss'
    },
    {
        'question': 'What does the `torchvision.models` module provide in PyTorch?',
        'options': 'Pre-trained models,Activation functions,Optimizers,Data loaders',
        'correct_answer': 'Pre-trained models'
    },
    {
        'question': 'In PyTorch, what is the purpose of the Batch Normalization layer in a neural network?',
        'options': 'Improves convergence during training,Reduces the number of parameters,Adds non-linearity to the model,Increases the learning rate',
        'correct_answer': 'Improves convergence during training'
    },
    {
        'question': 'How can you initialize the weights of a neural network in PyTorch?',
        'options': 'torch.init_weights(),torch.initialize_weights(),torch.nn.init(),torch.weights_init()',
        'correct_answer': 'torch.nn.init()'
    },
    {
        'question': 'Which function is used to perform a forward pass in PyTorch?',
        'options': 'model.forward(),model.backward(),model.predict(),model.training()',
        'correct_answer': 'model.forward()'
    },
    {
        'question': 'What is the purpose of the Rectified Linear Unit (ReLU) activation function?',
        'options': 'Introduces non-linearity,Ensures smooth gradients,Reduces overfitting,Normalizes the input',
        'correct_answer': 'Introduces non-linearity'
    },
    {
        'question': 'How do you define a convolutional layer in PyTorch?',
        'options': 'torch.FullyConnectedLayer(),torch.Conv2d(),torch.MaxPooling2d(),torch.ActivationLayer()',
        'correct_answer': 'torch.Conv2d()'
    },
    {
        'question': 'What is the purpose of the `torchvision.transforms` module?',
        'options': 'Applies transformations to images,Defines loss functions,Optimizes model parameters,Handles automatic differentiation',
        'correct_answer': 'Applies transformations to images'
    },
    {
        'question': 'Which optimizer is commonly used for training deep neural networks in PyTorch?',
        'options': 'Adam,Stochastic Gradient Descent (SGD),RMSprop,Adagrad',
        'correct_answer': 'Adam'
    },
    {
        'question': 'How do you perform transfer learning in PyTorch using a pre-trained model?',
        'options': 'Load the pre-trained model and replace the final fully-connected layer,Load the pre-trained model and replace all layers,Train the model from scratch,Use the pre-trained model without any modifications',
        'correct_answer': 'Load the pre-trained model and replace the final fully-connected layer'
    },
    {
        'question': 'What is the purpose of the `torch.nn.CrossEntropyLoss` in PyTorch?',
        'options': 'Combines softmax activation and negative log-likelihood loss,Calculates mean squared error loss,Applies binary cross-entropy loss,Defines a custom loss function',
        'correct_answer': 'Combines softmax activation and negative log-likelihood loss'
    },
    {
        'question': 'In PyTorch, how do you calculate the number of parameters in a neural network model?',
        'options': 'Sum the number of parameters in each layer,Count the number of layers,Use the `torch.nn.parameter_count()` function,It is not possible to calculate the number of parameters',
        'correct_answer': 'Sum the number of parameters in each layer'
    },    
        ]

# Create a DataFrame
pytorch_df = pd.DataFrame(pytorch_quiz_data)

# Write DataFrame to CSV file
pytorch_df.to_csv('pytorch_quiz_data.csv', index=False)
