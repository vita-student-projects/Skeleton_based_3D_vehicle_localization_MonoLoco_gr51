Group 51: Skeleton Based 3D Vehicle Localization

In this project, since there was no method specific for cars, we are inspired by the article MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation and repository: https://github.com/vita-epfl/monoloco. We tried to alter this model for cars.

Dataset: Kitti 3D object dataset which contains 7481 training and 7518 test images with labels.  

Method:
    First, we used OpenPifPaf's sparse model to extract 2D joints of cars from images which will be the baseline for the inputs of our MonoLoco model. 
    Those keypoints are preprocessed for our model, created a single json file like the MonoLoco's approach
    This data is fed into RNN of altered MonoLoco model.*
    The outputs are visualized by the program as well.*


Our contribution: 
    Model has been altered for being able to take higher dimension input.
    Data Preprocessing stage is redone:
        Adopted human body parts for cars (head: roof of the car, shoulders: mirrors of the car etc.)
        Average human sizes are changed to average car sizes, male/female distinction is used for SUV/regular family car difference, reference car is selected as a Toyota Corolla
        Transforms are redone for cars, Coco Keypoints are taken as a reference.
        Every other human/cyclist specific part is changed to car data of KITTI dataset
    Visualization part is altered for displaying cars:
        Male/Female distinction is again used for SUV/regular family car distinction. Used child distinction for identifying smaller cars, Mercedes A/E Station, GLA, GLE are taken as reference.
        By changing the dimensions, output now looks like a car-like prisma instead of stick-like human being (not tested)
    Since many parts of the code are shared, changes are done for MonoStereo approach as well.

Results: 
    Since training stage is not completed, we cannot show any numbers.
    We could only get the json files for training and testing (processed skeletons)

Conclusion:
    Due to limited time and resources, combined with underestimating the number of changes/problems in the preprocessing step, we cannot conclude the project. However, we believe that our steps are quite important for expanding MonoLoco to vehicles. After training and tuning some of the hyperparameters, there is a possibility that we can see a reasonable result for "first implementation".

*Could not run/test those parts