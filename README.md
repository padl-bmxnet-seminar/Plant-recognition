# Mushroom recognition 

## Dataset
The networks were trained with a diverse dataset of mushrooms photographed from diﬀerent angles, separately or in groups with variable sizes and backgrounds. For the used dataset, we combined two sets of mushroom images: The ﬁrst one is from a recent Kaggle challenge and includes pictures of nine genera of the most 
common Northern European mushrooms including 300 to 1500 selected images for each class. It was originally provided by the mycologist’s society of Northern Europe (https://www.kaggle.com/maysee/mushrooms-classiﬁcation-common-genuss-images).
This set was enriched by image data acquired by Svampe Atlas (https://svampe.databasen.org/), a Danish group carefully maintaining a database containing over 100000 fungi images. The Svampe Atlas collected a comprehensive representation 
of nearly 1500 wild mushrooms species, which have been spotted and photographed by the general public in Denmark. The Svampe Atlas data used in this project is from an image set which was created 
for a competition at the FGVC5 (ﬁne-grained visual-categorization) workshop at CVPR6 in 2018. The various species were merged with their corresponding genera if existing in the nine classes of the ﬁrst dataset. 
The training scripts also include code to import other datasets we used for testing before.

## Training 
With binary_training.py you are able to train a Binary Neural Network. With fit.py and run.py you can train a similar full-precision model, to create a comparable baseline. 
Furthermore you can load pretrained models to go on training with their parameters, to observe the effects of transfer learning in both binary and full-precision training.


