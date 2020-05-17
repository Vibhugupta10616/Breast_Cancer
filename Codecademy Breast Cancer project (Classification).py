import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

# To get info about the data
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)
#print(breast_cancer_data.target)
#print(breast_cancer_data.target_names)

#Splitting the data
training_data, validation_data , training_labels, validation_labels = train_test_split(breast_cancer_data.data,breast_cancer_data.target, test_size = 0.2 , random_state = 6)

#Finding the value of k for highest accuracy
#k_range = range(1,101)
#accuracies = []
#for i in k_range:
  #classifier = KNeighborsClassifier(i)
  #classifier.fit(training_data , training_labels)
  #accuracies.append(classifier.score(validation_data , validation_labels))

#plt.plot(k_range,accuracies)
#plt.xlabel("K Values")
#plt.title("Breast Cancer Classifier Accuracy")

#plt.show()

#predicting the score
classifier = KNeighborsClassifier(20)
classifier.fit(training_data , training_labels)
print(classifier.score(validation_data , validation_labels))

#Write your datapoint to predict




