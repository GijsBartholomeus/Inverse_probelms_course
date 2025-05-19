# Inverse_probelms_course
A project about the reconstruction of medical images.

## Project Structure

The idea is to first seek the right parameters such that we have two methods that work.
Algo A and Algo B --> return reconstructed images that are slightly different.
We do this testing in experiments.

Then we create a batch of reconstructed images both for algo A and B.
We then apply data augmentation and cutting in smaller patches.

We train a classifier to distinish between the two classes.