import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import keras.backend as K
import sys
import time
import matplotlib.pyplot as plt

from scipy import ndimage

np.set_printoptions(threshold=sys.maxsize, suppress=True)

class ConvolutionLayer:
    def __init__(self, KernelSize, NumberOfKernels, InputShape = None, Activation = None):

        self.KernelSize = KernelSize
        self.NumberOfKernels = NumberOfKernels
        self.InputShape = InputShape

        self.Activation = Activation

        self.Alpha = 0.001
        self.epsilon = 0.00000001
        self.Beta1 = 0.9
        self.Beta2 = 0.999

    def Config(self, InputShape):
        self.InputShape = InputShape
        self.OutputShape = (InputShape[0], self.InputShape[1] - self.KernelSize + 1, self.InputShape[2] - self.KernelSize + 1, self.NumberOfKernels)
        
        self.Kernels = np.random.randn(self.KernelSize, self.KernelSize, self.InputShape[3], self.NumberOfKernels) / (self.KernelSize * self.KernelSize * self.InputShape[3])
        #self.Kernels = np.arange(self.KernelSize * self.KernelSize * self.InputShape[3] * self.NumberOfKernels).reshape((self.KernelSize, self.KernelSize, self.InputShape[3], self.NumberOfKernels))
        self.Biases = np.random.randn(self.NumberOfKernels) / (self.NumberOfKernels)

        self.Momentum = np.zeros((self.KernelSize, self.KernelSize, self.InputShape[3], self.NumberOfKernels))
        self.Velocity = np.zeros((self.KernelSize, self.KernelSize, self.InputShape[3], self.NumberOfKernels))

        return self.OutputShape
    
    def Forward(self, Input, Inference = False):
        self.Input = Input

        Output = np.zeros((Input.shape[0], *self.OutputShape[1:]))

        # instead of looping over the pixels, i loop over the kernels as each pixel will be multiplied be each kernel element exept from at the edges.
        # i've done it this way as python loops are slow and this way i can rely on numpy's optimized functions
        for i in range(self.KernelSize):
            for j in range(self.KernelSize):
                Inputs = Input[:, i:i + self.OutputShape[1], j:j + self.OutputShape[2], :]  # trim the edges to only have the pixels that are to be multiplied by these kernels
                Kernel = self.Kernels[i, j, :, :]  # get this loop's kernel's weights
                Result = np.tensordot(Inputs, Kernel, axes = ((3), (0))) # multiply the inputs by the kernel's weights, 
                Output += Result # add the result to the output

        Output += self.Biases
        if self.Activation is not None: # apply any activation function
            Output = self.Activation.Forward(Output)
        return Output
    
    def Backward(self, Input, LearningRate = 0.01):
        Gradient = self.Activation.Backward(Input) 
        BiasesAdjustment = np.sum(Gradient.copy(), axis = (0, 1, 2)) / self.Input.shape[0] / self.Input.shape[1] / self.Input.shape[2] # calculate the adjustment to the biases
        WeightsAdjustment = np.zeros(self.Kernels.shape) # the adjustment to the kernels
        InputDerivs = np.zeros(self.InputShape) # the derivative of the input, this will be passed to the previous layer
        for i in range(self.KernelSize): # as stated in the forward function, i loop over the kernels as each pixel, exept from at the edges, are used in very similar calculations
            for j in range(self.KernelSize):
                Inputs = self.Input[:, i:i + self.OutputShape[1], j:j + self.OutputShape[2], :]
                WeightsAdjustment[i, j, :, :] = np.tensordot(Inputs, Gradient, axes = ((0, 1, 2), (0, 1, 2)))
                InputDerivs[:, i:i + self.OutputShape[1], j:j + self.OutputShape[2], :] += np.tensordot(Gradient, self.Kernels[i, j, :, :], axes = ((3), (1)))

        # calculate the momentum and velocity for adam optimization

        WeightsAdjustment = WeightsAdjustment / self.Input.shape[0]

        WeightMomentums = self.Beta1 * self.Momentum + (1 - self.Beta1) * WeightsAdjustment
        WeightVelocities = self.Beta2 * self.Velocity + (1 - self.Beta2) * np.square(WeightsAdjustment)

        self.Momentum = WeightMomentums
        self.Velocity = WeightVelocities

        WeightMomentumsCorrected = WeightMomentums / (1 - self.Beta1)
        WeightVelocitiesCorrected = WeightVelocities / (1 - self.Beta2)

        WeightsAdjustment = WeightMomentumsCorrected  * (self.Alpha / (np.sqrt(WeightVelocitiesCorrected) + self.epsilon))


        self.Kernels -= WeightsAdjustment

        # adam isn't used for biases so learning rate is used to stabilize the adjustment
        self.Biases -= BiasesAdjustment * LearningRate

        return InputDerivs
    
class AveragePoolingLayer:
    def __init__(self, KernelSize):
        self.KernelSize = KernelSize

    def Config(self, InputShape):

        self.InputShape = InputShape
        self.OutputShape = (InputShape[0], self.InputShape[1] // self.KernelSize, self.InputShape[2] // self.KernelSize, self.InputShape[3])

        return self.OutputShape
    
    def Forward(self, Input, Inference = False):

        # Pooling layer without any loops
        
        # I first shave off the edges of the input that do not fit in a kernel, so that the input can be reshaped into equal parts
        ShaveX = self.InputShape[1] % self.KernelSize
        ShaveY = self.InputShape[2] % self.KernelSize

        self.ShaveX = ShaveX
        self.ShaveY = ShaveY

        Input = Input[:, :self.InputShape[1] - ShaveX, :self.InputShape[2] - ShaveY, :]
        self.Shape = Input.shape

        InstancesX = self.OutputShape[1]
        InstancesY = self.OutputShape[2]

        SplitX = np.array(np.split(Input, InstancesX, axis = 1)) # split the x axis of the samples into kernel widths
        SplitX = np.moveaxis(SplitX, 0, 1) # move the axis so that the samples are the first axis
        SplitY = np.array(np.split(SplitX, InstancesY, axis = 3)) # split the y axis of the samples into kernel heights
        SplitY = np.moveaxis(SplitY, 0, 2)

        Average = np.average(SplitY, axis = (3, 4)) # calculate the average of all kernels at once
        self.Average = Average

        return Average

    def Backward(self, DerivsAverage, LearningRate = 0.01):
        # average pooling layer backpropagation is straight forward as the average is the same for all the pixels in the kernel and there aren't any weights
        # therefore the derivatives can just be expanded to kernel size and then reshaped to the input shape
        Derivs = np.tile(DerivsAverage, (1, 1, self.KernelSize, self.KernelSize)).reshape(self.Shape)
        Derivs = Derivs / (self.KernelSize * self.KernelSize)

        # if the edges were shaved off in the forward pass, they need to be added back here
        if self.ShaveX != 0 or self.ShaveY != 0:
            Derivs = np.pad(Derivs, ((0, 0), (0, self.ShaveX), (0, self.ShaveY), (0, 0)), 'constant', constant_values = 0) # as they were removed they had no effect on the output so they are just zeros

        return Derivs

class PoolingLayer:
    def __init__(self, KernelSize):
        self.KernelSize = KernelSize

    def Config(self, InputShape):
        self.InputShape = InputShape
        self.OutputShape = (InputShape[0], self.InputShape[1] // self.KernelSize, self.InputShape[2] // self.KernelSize, self.InputShape[3])

        return self.OutputShape
    
    def Forward(self, Input, Inference = False):

        ShaveX = self.InputShape[1] % self.KernelSize
        ShaveY = self.InputShape[2] % self.KernelSize

        Input = Input[:, :self.InputShape[1] - ShaveX, :self.InputShape[2] - ShaveY, :]

        InstancesX = self.OutputShape[1]
        InstancesY = self.OutputShape[2]

        # like the average pooling, the input is split into the kernels
        SplitX = np.array(np.split(Input, InstancesX, axis = 1))
        SplitX = np.moveaxis(SplitX, 0, 1)
        SplitY = np.array(np.split(SplitX, InstancesY, axis = 3))
        SplitY = np.moveaxis(SplitY, 0, 2)
        # the max is then taken of each kernel
        Max = np.max(SplitY, axis = (3, 4))

        # a map is made of the max values for use in the backpropagation
        if not Inference: # if it is inference, the map is not needed as it won't execute the backpropagation algorithm
            # flatten the kernels so that the max can be found this is because numpy's argmax can only find the max of a 1d array
            Flat = SplitY.reshape((SplitY.shape[0], SplitY.shape[1], SplitY.shape[2], SplitY.shape[3] * SplitY.shape[4],  SplitY.shape[5])) 
            MapIndex = np.argmax(Flat, axis = 3).reshape(Flat.shape[0]* Flat.shape[1] * Flat.shape[2], Flat.shape[4]) # find the index of the max
            Map = np.zeros((Flat.shape[0] * Flat.shape[1] * Flat.shape[2], Flat.shape[3], Flat.shape[4])) # the undrawn map, with the kernels flattened
            Map[np.arange(Map.shape[0])[:, np.newaxis], MapIndex, np.arange(Map.shape[2])] = 1 # draw the map
            # reshape the "Map" to map the max values of the input
            Map = Map.reshape(Flat.shape)
            Map = np.array(np.split(Map, 2, axis=3))
            Map = np.moveaxis(Map, 0, 2)

            Map = Map.reshape(*Input.shape)

            # the shaves only need to be saved for the backpropagation
            self.ShaveX = ShaveX
            self.ShaveY = ShaveY

            self.Map = Map

        return Max

    def Backward(self, Input, LearningRate = 0.01):
        # tile the derivatives so that the derivative of the kernel's output matches to the kernel's inputs
        Derivs = np.tile(Input, (1, 1, self.KernelSize, self.KernelSize)).reshape(self.Map.shape)
        Derivatives = Derivs * self.Map # multiply the derivatives by the map so if the input wasn't the max it is multiplied by 0 and if it was it is multiplied by 1
        if self.ShaveX != 0 or self.ShaveY != 0:
            Derivatives = np.pad(Derivatives, ((0, 0), (0, self.ShaveX), (0, self.ShaveY), (0, 0)), 'constant', constant_values = 0)
        return  Derivatives

class FlattenLayer:
    def __init__(self, InputShape = None):
        pass

    def Config(self, InputShape):
        self.InputShape = InputShape
        self.OutputShape = (InputShape[0], InputShape[1] * InputShape[2] * InputShape[3])

        return self.OutputShape
    
    def Forward(self, Input, Inference = False):
        return Input.reshape(Input.shape[0], *self.OutputShape[1:]) # simply reshape the input to the (flat) output shape

    def Backward(self, Input, LearningRate = 0.01):
        return Input.reshape(self.InputShape) # reshape the derivatives to the input shape

class FullyConnectedLayer: # straight forward implementation of a fully connected layer, with adam optimization
    def __init__(self, InputShape, OutputShape, ActivationFunction = None):
        self.InputShape = InputShape
        self.OutputShape = OutputShape
        self.ActivationFunction = ActivationFunction

        self.Alpha = 0.001
        self.epsilon =  0.00000001
        self.Beta1 = 0.9
        self.Beta2 = 0.999

    def Config(self, InputShape):
        self.InputShape = InputShape
        self.OutputShape = (InputShape[0], self.OutputShape)

        self.Weights = (np.random.randn(self.InputShape[1], self.OutputShape[1]) - 0.5) / (self.InputShape[1])
        self.Biases = np.random.randn(self.OutputShape[1]) / self.OutputShape[1]

        self.Momentum = np.zeros(self.Weights.shape)
        self.Velocity = np.zeros(self.Weights.shape)

        return self.OutputShape
    
    def Forward(self, Input, Inference = False):
        self.Input = Input
        Output = np.dot(Input, self.Weights) + self.Biases
        if self.ActivationFunction is not None:
            Output = self.ActivationFunction.Forward(Output)
        return Output

    def Backward(self, Input, LearningRate = 0.01):
        if self.ActivationFunction is not None:
            Input = self.ActivationFunction.Backward(Input)

        BiasesAdjustment = np.sum(Input.copy(), axis = 0) / self.Input.shape[0]
        WeightsAdjustment = np.dot(self.Input.T, Input) / self.Input.shape[0]

        WeightMomentums = self.Beta1 * self.Momentum + (1 - self.Beta1) * WeightsAdjustment
        WeightVelocities = self.Beta2 * self.Velocity + (1 - self.Beta2) * np.square(WeightsAdjustment)

        self.Momentum = WeightMomentums
        self.Velocity = WeightVelocities

        WeightMomentumsCorrected = WeightMomentums / (1 - self.Beta1)
        WeightVelocitiesCorrected = WeightVelocities / (1 - self.Beta2)

        WeightsAdjustment = WeightMomentumsCorrected  * (self.Alpha / (np.sqrt(WeightVelocitiesCorrected) + self.epsilon))

        InputDerivative = np.dot(Input, self.Weights.T)

        self.Weights -= WeightsAdjustment
        self.Biases -= BiasesAdjustment * LearningRate
        
        return InputDerivative
    
class BatchNormalizationLayer:
    def __init__(self, InputShape = None):

        self.InputShape = InputShape
        self.OutputShape = InputShape

        self.Alpha = 0.001
        self.epsilon = 0.00000001
        self.Beta1 = 0.9
        self.Beta2 = 0.999

    def Config(self, InputShape):
        self.InputShape = InputShape
        self.OutputShape = InputShape

        self.Gamma = np.random.randn(self.InputShape[-1])
        self.Beta = np.random.randn(self.InputShape[-1])

        self.Momentum = np.zeros(self.Gamma.shape)
        self.Velocity = np.zeros(self.Gamma.shape)

        self.RunningMean = np.zeros(self.InputShape[-1])
        self.RunningVariance = np.zeros(self.InputShape[-1])

        self.SumAxis = tuple(range(len(self.InputShape) - 1))

        return self.OutputShape
    
    def Forward(self, Input, Inference = False):


        self.Input = Input

        if not Inference: # if it is inference, the running mean and variance are used, as that is what the model has learned

            self.Mean = np.mean(Input, axis = 0)
            self.Variance = np.var(Input, axis = 0)

            self.RunningMean = self.Beta1 * self.RunningMean + (1 - self.Beta1) * self.Mean
            self.RunningVariance = self.Beta1 * self.RunningVariance + (1 - self.Beta1) * self.Variance

        else:
            self.Mean = self.RunningMean
            self.Variance = self.RunningVariance

        
        # normalize the input
        XMu = Input - self.Mean
        self.XMu = XMu.copy()

        StandardDeviation = np.sqrt(self.Variance + self.epsilon)
        InverseStandardDeviation = 1 / StandardDeviation

        self.InverseStandardDeviation = InverseStandardDeviation

        self.Normalized = XMu * InverseStandardDeviation.copy()

        Output = self.Gamma * self.Normalized + self.Beta

        return Output
    
    def Backward(self, Gradient, LearningRate = 0.01):
        # calculate the derivatives of the input, gamma and beta
        # very maths heavy, but it is just the chain rule applied to the batch normalization function
        DXhat = Gradient * self.Gamma 
        DVar = np.sum(DXhat * self.XMu, axis = 0) * -0.5 * self.InverseStandardDeviation ** 3  # the derivative of the variance
        DMu = np.sum(DXhat * -self.InverseStandardDeviation, axis = 0) + DVar * np.mean(-2 * self.XMu, axis = 0) # the derivative of the mean
        DX = (DXhat * self.InverseStandardDeviation) + (DVar * 2 * self.XMu / self.Input.shape[0]) + (DMu / self.Input.shape[0]) # the derivative of the input

        DGamma = np.sum(Gradient * self.Normalized, axis = self.SumAxis)
        DBeta = np.average(Gradient, axis = self.SumAxis) 

        # adam optimization
        GammaMomentums = self.Beta1 * self.Momentum + (1 - self.Beta1) * DGamma
        GammaVelocities = self.Beta2 * self.Velocity + (1 - self.Beta2) * np.square(DGamma)

        self.Momentum = GammaMomentums
        self.Velocity = GammaVelocities

        GammaMomentumsCorrected = GammaMomentums / (1 - self.Beta1)
        GammaVelocitiesCorrected = GammaVelocities / (1 - self.Beta2)

        GammaAdjustment = GammaMomentumsCorrected  * (self.Alpha / (np.sqrt(GammaVelocitiesCorrected) + self.epsilon))

        self.Gamma -= GammaAdjustment
        self.Beta -= DBeta * self.Alpha

        return DX
    

class DropoutLayer:
    def __init__(self, Probability):
        self.Probability = Probability

    def Config(self, InputShape):
        self.InputShape = InputShape
        self.OutputShape = InputShape
        return self.OutputShape

    def Forward(self, Input, Inference = False):
        if Inference: # if its testing, all the neurons are used
            return Input 
        self.Mask = np.random.binomial(1, self.Probability, size = Input.shape) # create a mask with Probabilty chance of being 1
        Input[self.Mask == 0] = 0 # set the neurons to 0 if the mask is 0
        return Input
    
    def Backward(self, Input, LearningRate = 0.01):
        Input[self.Mask == 0] = 0 # the neurons that were dropped have no effect on the output so they're derivatives are 0
        return Input
        
class SoftmaxLayer: # Simple softmax implementation
    def __init__(self):
        pass

    def Forward(self, Input):
        Exp = np.exp(Input)
        SumExp = np.sum(Exp, axis=1)
        Result = Exp / SumExp[:, np.newaxis]
        return Result

    def Backward(self, Input):
        return Input
        
class ReLULayer: # Simple ReLU implementation
    def __init__(self):
        pass

    def Forward(self, Input):
        self.Z = Input
        return np.maximum(0, Input)

    def Backward(self, Input): 
        Input[self.Z <= 0] = 0
        return Input

class LeakyReLULayer: # Simple Leaky ReLU implementation
    def __init__(self, Alpha = 0.01):
        self.Alpha = Alpha

    def Forward(self, Input):
        self.Z = Input
        Input[self.Z <= 0] *= self.Alpha
        return Input

    def Backward(self, Input): 
        Input[self.Z <= 0] *= self.Alpha 
        return Input

class Network:
    # design the network here
    def __init__(self, BatchSize, InputShape = None, OutputSize = 10, DataWrapper = None, NonLinearOutputMap = False):
        self.OutputSize = OutputSize
        self.BatchSize = BatchSize
        self.Layers = [  # example network
            ConvolutionLayer(3, 16, InputShape = InputShape, Activation = LeakyReLULayer()),
            ConvolutionLayer(3, 16, Activation = LeakyReLULayer()),
            AveragePoolingLayer(2),
            #ConvolutionLayer(3, 32, Activation = LeakyReLULayer()),
            #ConvolutionLayer(3, 64, Activation = LeakyReLULayer()),
            FlattenLayer(),
            BatchNormalizationLayer(),
            #DropoutLayer(0.9),
            FullyConnectedLayer(0, 128, ActivationFunction = LeakyReLULayer()),
            FullyConnectedLayer(0, OutputSize, ActivationFunction = SoftmaxLayer())
        ]
        for Index in range (len(self.Layers)):
            Layer = self.Layers[Index]
            InputShape = Layer.Config(InputShape)

        self.OutputMap = np.arange(self.OutputSize)

    def Forward(self, Input, Inference = False):
        for Layer in self.Layers: # forward pass
            # this basic implementation is why many layers have redundant Inferece parameters, as i felt it was easier to just pass it through, 
            # then to check if the layer is an instance of a certain class
            Input = Layer.Forward(Input, Inference = Inference)
        return Input

    def Backward(self, LearningRate = 0.01, Gradient = None):
        for Layer in reversed(self.Layers): # backward pass
            Gradient = Layer.Backward(Gradient, LearningRate)
        return Gradient

    def Encode(self, Labels): # simple one hot encoding
        Encoded = np.zeros((Labels.size, Labels.max() + 1))
        Encoded[np.arange(Labels.size), Labels] = 1 
        return Encoded
    
    def Decode(self, Encoded): # simply finding the max value of the predictions
        ArgMax = np.argmax(Encoded, axis = 1)
        return self.OutputMap[ArgMax]
    
    def GetAccuracy(self, Input, Labels):
        Output = self.Decode(Input)
        return np.sum(Output == Labels)
    
    def Train(self, LearningRate = 0.01, DataWrapper = None): # training function
        DataWrapper.Shuffle()
        DataWrapper.ResetBatchPosition()
        Iterations = DataWrapper.GetIterations(self.BatchSize) # get the number of iterations needed to go through all the batches of the dataset
        Accuracy, Loss = 0, 0
        Start = time.time() # started really getting into optimizing the code so i started timing it

        for Index in range(0, Iterations): # training loop
            print("", end="\r")
            print("Iteration: ", (Index/Iterations) * 100, "%", end="\r") # print the progress of the training
            BatchInput, BatchLables, BatchEncoded= DataWrapper.GetBatch(self.BatchSize)

            Output = self.Forward(BatchInput)
            Accuracy += self.GetAccuracy(Output, BatchLables)

            Gradient = (Output - BatchEncoded)
            Cost = -np.sum(BatchEncoded * np.log(Output + 0.00000001), axis = 1)
            TotalCost = np.sum(Cost)
            Loss += TotalCost
            self.Backward(LearningRate, Gradient = Gradient)

        print("Training:  Accuracy: ", (Accuracy/(self.BatchSize*Iterations)) * 100, "%, Loss: ", Loss / (self.BatchSize*Iterations), "Time: ", time.time() - Start)

        return Accuracy
    
    def Test(self, Input, Labels):
        # the entire test set is passed at once, generally this is not a good idea as it can be very memory intensive
        Output = self.Forward(Input, Inference = True) # inference flags that the model is not training so some layers have different behaviour
        Accuracy = self.GetAccuracy(Output, Labels)
        BatchEncoded = self.Encode(Labels)
        loss = -np.sum(BatchEncoded * np.log(Output + 0.00000001))
        print("Testing: Accuracy: ", (Accuracy/Labels.size) * 100, "%, Loss: ", loss)

    def GenerateSubmition(self, Input): # just a function to generate a file for the kaggle competition
        Output = self.Forward(Input, Inference = True)
        Prediction = self.Decode(Output)
        Ids = np.arange(1, Prediction.size + 1)
        Data = np.stack((Ids, Prediction), axis = 1)
        np.savetxt("Submition.csv", Data, delimiter = ",", header = "ImageId,Label", fmt = "%d", comments = "")
        return Output

class DataHandler: # i created the data handler to manage data preprocessing and loading
    def __init__(self): 
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # load the mnist dataset
        # normalize the data
        self.x_train = x_train.astype("float32") / 255.0 
        self.x_test = x_test.astype("float32") / 255.0

        # the network is designed to work with 4d data, so if the data is 3d it is expanded to 4d
        if len(self.x_train.shape) <= 3:
            self.x_train = np.expand_dims(self.x_train, -1)
            self.x_test = np.expand_dims(self.x_test, -1)

        if len(y_train.shape) > 1: 
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]

        self.y_train = y_train
        self.y_test = y_test

        self.EncodeMax = 0

        self.EncodedTrain = self.Encode(self.y_train)
        self.EncodedTest = self.Encode(self.y_test)

        self.BatchPosition = 0
        self.Epoch = 0

        self.NoClasses = 10

    def Encode(self, Labels): # simple one hot encoding
        self.EncodeMax = max(Labels.max(), self.EncodeMax)
        Encoded = np.zeros((Labels.size, self.EncodeMax + 1))
        Encoded[np.arange(Labels.size), Labels] = 1 
        return Encoded

    def Shuffle(self): # shuffle the data as it has been proven to imporve the training
        Permutation = np.random.permutation(self.x_train.shape[0])
        self.x_train = self.x_train[Permutation]
        self.y_train = self.y_train[Permutation]
        self.EncodedTrain = self.EncodedTrain[Permutation]

    def GetBatch(self, BatchSize): # get a batch of data
        self.BatchSize = BatchSize
        if self.BatchPosition + BatchSize < self.x_train.shape[0]:
            self.BatchPosition += BatchSize
            return self.x_train[self.BatchPosition - BatchSize:self.BatchPosition], self.y_train[self.BatchPosition - BatchSize:self.BatchPosition], self.EncodedTrain[self.BatchPosition - BatchSize:self.BatchPosition]
        return self.x_train[:BatchSize], self.y_train[:BatchSize], self.EncodedTrain[:BatchSize]
    
    def ResetBatchPosition(self): # reset the batch position and increase the epoch
        self.BatchPosition = 0
        self.Epoch += 1

    def GetIterations(self, BatchSize): # get the number of iterations needed to go through all the batches of the dataset
        return self.x_train.shape[0] // BatchSize
    
    def GenerateInputShape(self): # generate the input shape for the network based on the dataset
        return self.x_train.shape[1:]

    def CreateModifiedData(self, Input, Labels): # used for data augmentation
        Transformations = np.random.randint(0, 4, (Input.shape[0])) # randomly select a transformation for each image

        # get a map of which images are to be transformed by which transformation
        TransformByTranslation = Transformations == 0
        TransformByRotation = Transformations == 1
        TransformByScaling = Transformations == 2
        TransformByNoise = Transformations == 3

        # execute the selected transformations
        Translated = self.Shift(Input[TransformByTranslation])
        Rotated = self.Rotate(Input[TransformByRotation])
        Scaled = self.Scale(Input[TransformByScaling])
        Noised = self.AddNoise(Input[TransformByNoise])

        # combine the transformed images, ensuring that the labels are also combined in the same order
        Transformed = np.concatenate((Translated, Rotated, Scaled, Noised), axis = 0)
        TransformedLabels = np.concatenate((Labels[TransformByTranslation], Labels[TransformByRotation], Labels[TransformByScaling], Labels[TransformByNoise]), axis = 0)

        TransforedLabelsEncoded = self.Encode(TransformedLabels)

        return Transformed, TransformedLabels, TransforedLabelsEncoded
    
    def Shift(self, Input):
        if Input.shape[0] == 0: # as random is used, the input can be empty
            return Input
        Output = np.zeros(Input.shape)
        for i in range(Input.shape[0]): # i couldn't find a way to do it randomly for all images at once, so i loop over each image
            # shift the image by a random amount, using ndimage.shift
            ShiftX = np.random.randint(-2, 3)
            ShiftY = np.random.randint(-2, 3)
            Output[i] = ndimage.shift(Input[i], (ShiftX, ShiftY), cval = 0)
        return Output

    def Rotate(self, Input):
        if Input.shape[0] == 0:
            return Input
        Output = ndimage.rotate(Input[0], np.random.randint(-15, 16), reshape = False) # i decided to build the output instead of overriting an empty array, as it uses less memory
        for i in range(1, Input.shape[0]):
            Rotation = np.random.randint(-15, 16)
            NewImage = ndimage.rotate(Input[i], Rotation, reshape = False)
            Output = np.concatenate((Output, NewImage), axis = 0)

        return Output

    def Scale(self, Input):
        #pad each image with zeros by a random amount or crop each image by a random amount
        # then resize the image to the original size
        if Input.shape[0] == 0:
            return Input
        InputX = Input.shape[1]
        InputY = Input.shape[2]

        PaddX = np.random.randint(0, 3)
        PaddY = np.random.randint(0, 3)
        PaddedFirst = np.pad(Input[0], ((PaddX, PaddX), (PaddY, PaddY)), 'constant', constant_values = 0)
        Output = cv2.resize(PaddedFirst, (InputX, InputY), interpolation = cv2.INTER_AREA)
        for i in range(1, Input.shape[0]):
            ScaleX = np.random.randint(-1, 3)
            ScaleY = np.random.randint(-1, 3)
            NewImage = Input[i]
            if ScaleX < 0:
                NewImage = NewImage[-ScaleX:InputX+ScaleX, :]
            else:
                NewImage = np.pad(NewImage, ((ScaleX, ScaleX), (0, 0)), 'constant', constant_values = 0)
            if ScaleY < 0:
                NewImage = NewImage[:, -ScaleY:InputY+ScaleY]
            else:
                NewImage = np.pad(NewImage, ((0, 0), (ScaleY, ScaleY)), 'constant', constant_values = 0)

            NewImageFixed = cv2.resize(NewImage, (InputX, InputY), interpolation = cv2.INTER_AREA)
            Output = np.concatenate((Output, NewImageFixed), axis = 0)

        return Output

    def AddNoise(self, Input):
        if Input.shape[0] == 0:
            return Input
        # this one is nice and easy as numpy has a function for it
        Noise = np.random.normal(0, 0.05, Input.shape)
        return Input + Noise


# implementing and using the classes
print("Starting")

DataWrapper = DataHandler()

batch_size = 128
input_shape = (batch_size, *DataWrapper.GenerateInputShape())

CNN = Network(batch_size, input_shape, OutputSize = 10, DataWrapper = DataWrapper)
HighestAccuracy = 0

for i in range(100):
    print("Epoch: ", i + 1)
    Accuracy = CNN.Train(DataWrapper = DataWrapper, LearningRate = 0.001)
    CNN.Test(DataWrapper.x_test, DataWrapper.y_test)
    print()