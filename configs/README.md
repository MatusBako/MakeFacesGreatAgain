# Configuration file format

### Model configuration

When not training GAN, training configuration has section `CNN` with these fields:

- **OutputFolder <str>:** path to directory, where directory with results will be created

- **BatchSize <int>**

- **Device <'cuda', 'cpu'>**: one of supplied options or any valid device which can be supplied to `torch.device()`

- **IterationLimit <int>**

- **LearningRate <float>**

- **UpscaleFactor <int>**: different models support different factors, all of them support powers of 2

- **ModelName <str>:** name of the model from `models` directory

- **IterationsPerSnapshot <int>:** number if iterations after which a snapshot is created

- **IterationsPerImage <int>:** number if iterations after which outputs are saved

- **IterationsToEvaluation <int>:** number if iterations after which evaluation is run on test dataset

- **EvaluationIterations <int>:** number of iteration run during evaluation
  
  

When training GAN, the section name changes from `CNN` to `GAN`, **LearningRate** becomes **GeneratorLearningRate** and **DiscriminatorLearningRate**, **ModelName** becomes **Generator** and **Discriminator**. The config file GANs also contain scaling parameter for each of their lossess.



### Dataset configuration

Section `Dataset` contains following fields:

- **Class <str>:** name of class encapsulating the dataset from `datasets` directory

- **TrainData <str>:** path to directory containing training data

- **TestData <str>:** path to directory containing testing data

- **TrainLength <int>:** number of images used from training dataset

- **TestLength <int>:** number of images used from testing dataset



### Optimizers and schedulers

Both optimizer and have the same theree fields:

- **Name <str>:** name of class

- **Args <list>:** arguments

- **Kwargs <dict>:** keyword arguments

Class is imported dynamically and arguments are then supplied, so refer to `torch` documentation for valid values.




