##python code:

'''
Clustering students based on responses on the questionare
This data set contains a total 5820 evaluation scores provided by students from Gazi University in Ankara (Turkey). 
There is a total of 28 course specific questions and additional 5 attributes.

Attribute Information:
instr: Instructor's identifier; values taken from {1,2,3}
class: Course code (descriptor); values taken from {1-13}
repeat: Number of times the student is taking this course; values taken from {0,1,2,3,...}
attendance: Code of the level of attendance; values from {0, 1, 2, 3, 4}
difficulty: Level of difficulty of the course as perceived by the student; values taken from {1,2,3,4,5}
Q1: The semester course content, teaching method and evaluation system were provided at the start.
Q2: The course aims and objectives were clearly stated at the beginning of the period.
Q3: The course was worth the amount of credit assigned to it.
Q4: The course was taught according to the syllabus announced on the first day of class.
Q5: The class discussions, homework assignments, applications and studies were satisfactory.
Q6: The textbook and other courses resources were sufficient and up to date.
Q7: The course allowed field work, applications, laboratory, discussion and other studies.
Q8: The quizzes, assignments, projects and exams contributed to helping the learning.
Q9: I greatly enjoyed the class and was eager to actively participate during the lectures.
Q10: My initial expectations about the course were met at the end of the period or year.
Q11: The course was relevant and beneficial to my professional development.
Q12: The course helped me look at life and the world with a new perspective.
Q13: The Instructor's knowledge was relevant and up to date.
Q14: The Instructor came prepared for classes.
Q15: The Instructor taught in accordance with the announced lesson plan.
Q16: The Instructor was committed to the course and was understandable.
Q17: The Instructor arrived on time for classes.
Q18: The Instructor has a smooth and easy to follow delivery/speech.
Q19: The Instructor made effective use of class hours.
Q20: The Instructor explained the course and was eager to be helpful to students.
Q21: The Instructor demonstrated a positive approach to students.
Q22: The Instructor was open and respectful of the views of students about the course.
Q23: The Instructor encouraged participation in the course.
Q24: The Instructor gave relevant homework assignments/projects, and helped/guided students.
Q25: The Instructor responded to questions about the course inside and outside of the course.
Q26: The Instructor's evaluation system (midterm and final questions, projects, assignments, etc.) effectively measured the course objectives.
Q27: The Instructor provided solutions to exams and discussed them with students.
Q28: The Instructor treated all students in a right and objective manner.

Q1-Q28 are all Likert-type, meaning that the values are taken from {1,2,3,4,5}


'''

#!/usr/bin/env python
# coding: utf-8
##Unsupervised learning problem
##importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


#loading the dataset
dataset = pd.read_csv('C:\\Users\\Palash\\Downloads\\turkiye-student-evaluation_generic.csv')    


#checking the file 
dataset.head(3)

#checking dimensions
dataset.shape


#correlation between independent variables
plt.figure(figsize=(25,15))
sns.heatmap(dataset.corr(), annot=True)


#counting various responses
plt.figure(figsize = (20,6))
sns.countplot(x = 'class',data=dataset)

##Graph to see how the rating has been given by students for each question
'''
From the graph, we can see that very less students have given completely disagree (Rating 1) for Question Q14, Q15, Q17, Q19 - Q22, Q25By above graph, we can see that very less students have given completely disagree (Rating 1) for Question Q14, Q15, Q17, Q19 - Q22, Q25By above graph, we can see that very less students 
have given completely disagree (Rating 1) for Question Q14, Q15, Q17, Q19 - Q22, Q25

'''
plt.figure(figsize = (20,20))
sns.boxplot(data = dataset.iloc[:,5:31])

'''Lets understand the students have responded for questions against the classes
by calculating the mean for each question response for all the classes'''

questionmeans = []
classlist = []
questions = []

totalplotdata = pd.DataFrame(list(zip(classlist,questions,questionmeans)),columns = ['class','questions','mean'])
for class_num in range(1,13):
    class_data = dataset[(dataset['class']==class_num)]
    
    questionmeans = []
    classlist = []
    questions = []
    
    for num in range(1,13):
        questions.append(num)
    #Class related questions are from  Q1 to Q12
    for col in range(5,17):
        questionmeans.append(class_data.iloc[:,col].mean())
    classlist += 12 * [class_num]
    print(classlist)
    plotdata = pd.DataFrame(list(zip(classlist,questions,questionmeans)),columns = ['class','questions','mean'])
    totalplotdata = totalplotdata.append(plotdata,ignore_index = True)    


##Plotting mean rating against various classes
'''Graph shows that we have best ratings from Class 2 and worst rateing from class 4 students
'''
plt.figure(figsize = (20,10))
sns.pointplot(x = 'questions', y = 'mean',data = totalplotdata,hue = 'class')

'''Lets see how the rating has been given against instructor wise
by Calculating mean for each question response for all the classes'''
questionmeans = []
inslist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(inslist,questions,questionmeans)),columns = ['ins','questions','mean'])
for ins_num in range(1,4):
    ins_data = dataset[(dataset['instr'] == ins_num)]
    questionmeans = []
    inslist = []
    questions = []
    
    for num in range(13,29):
        questions.append(num)
        
    for col in range(17,33):
        questionmeans.append(ins_data.iloc[:,col].mean())
    inslist += 16 * [ins_num]
    plotdata = pd.DataFrame(list(zip(inslist,questions,questionmeans))
                            ,columns = ['ins','questions','mean'])
    totalplotdata = totalplotdata.append(plotdata,ignore_index = True)

    
##Plotting the rating against each instructor by taking the mean reponse for all the questions  
'''
Based on the graph we can see that According to the Student ratings we see that Instructor 1 and 2 
are performing well and got similar rateings but Instructor 3 got less ratings. 
So we can further explore which course instructor 3 teaches and find out the which course got least ratings.
'''
plt.figure(figsize = (20,5))
sns.pointplot(x = 'questions', y ='mean',data = totalplotdata,hue = 'ins')


##Calculate mean for each question response for all the classes for Instructor 3
dataset_inst3 = dataset[(dataset['instr'] == 3)]
class_array_for_inst3 = dataset_inst3['class'].unique().tolist()
questionmeans = []
classlist = []
questions = []

totalplotdata = pd.DataFrame(list(zip(classlist,questions,questionmeans))
                            ,columns =['class','questions','mean'])
for class_num in class_array_for_inst3:
    class_data = dataset_inst3[(dataset_inst3['class']==class_num)]
    
    questionmeans = []
    classlist = []
    questions = []
    
    for num in range(1,13):
        questions.append(num)
        
    for col in range(5,17):
        questionmeans.append(class_data.iloc[:,col].mean())
    classlist += 12* [class_num]
    
    plotdata = pd.DataFrame(list(zip(classlist,questions,questionmeans)),columns= ['class','questions','mean'])
    totalplotdata = totalplotdata.append(plotdata,ignore_index = True)
    
##PLottig theresponse for instructor 3
plt.figure(figsize = (20,8))
sns.pointplot(x = 'questions',y = 'mean',data = totalplotdata,hue = 'class') 



##Clustering students based on questionare data
dataset_questions = dataset.iloc[:,5:33]
dataset_questions.head(5)


##lets do a pca for feature dimension reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(dataset_questions)


##Plotting the graph to see WCSS(within cluster sum of squares)
from sklearn.cluster import KMeans
wcss = []
for i in range(1,7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_questions_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,7),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[19]:


##Based on elbow method we can go for 3 clusters
kmeans = KMeans(n_clusters = 3,init = 'k-means++')
y_kmeans = kmeans.fit_predict(dataset_questions_pca)


##Visualizing the clusters
##clusters_center_
plt.scatter(dataset_questions_pca[y_kmeans == 0,0],dataset_questions_pca[y_kmeans==0,1],s = 100,c = 'yellow',label = 'Cluster 1')

plt.scatter(dataset_questions_pca[y_kmeans == 1,0],dataset_questions_pca[y_kmeans == 1,1], s = 100, c = 'green', label = 'Cluster 2')

plt.scatter(dataset_questions_pca[y_kmeans == 2,0],dataset_questions_pca[y_kmeans == 2,1], s = 100, c = 'red', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 300, c = 'blue', label = 'Centroids')

plt.title('Clusters of students')

plt.xlabel('PCA 1')

plt.ylabel('PCA 2')

plt.legend()

plt.show()


##cluster of 3 students who have like negative,neutral and positive feedback 
##cheking the count of students in each cluster
import collections
collections.Counter(y_kmeans)


'''
So we have 2358 students who have given 
negative ratings overall , 2222 students with positive ratings and 1240 students with nuetral response

'''
##Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(dataset_questions_pca,method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('questions')
plt.ylabel('Euclidean distances')
plt.show()


##Fitting hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(dataset_questions_pca)
X = dataset_questions_pca

#Visualising the clusters
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s = 100, c = 'yellow',label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1,1],s = 100, c = 'red', label = 'Cluster 2')

plt.title('Clusters of STUDENTS')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

##Checking the count of students in each cluster
import collections
collections.Counter(y_hc)