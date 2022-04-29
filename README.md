# student-placement-analysis

 This project takes an analytical approach to students linking to their placement records and salaries. We have created a streamlit application to demonstrate our analysis and also integrated our ML model into the app to give our viewers the opportunity to tweak these supposed factors that affect the salary and view the outcome.

# Visualization 1

- Does gender bias exist in placements?

![Viz 1](https://github.com/kautilyak/student-placement-analysis/blob/main/viz/viz1.png)

# Visualization 2 
- Is there any coorelation between salary, degree percentage and specialization of students.

![Viz 2](https://github.com/kautilyak/student-placement-analysis/blob/main/viz/viz2.png)

# Visualization 3
- Do companies prefer candidates with work experience over the others?

![Viz 3](https://github.com/kautilyak/student-placement-analysis/blob/main/viz/viz3.png)

# Visualization 4 
- Average salaries of a degree and specialization combination:

![Viz 4](https://github.com/kautilyak/student-placement-analysis/blob/main/viz/viz4.png)

# Visualization 5
- How does work experience play a role in determining the candidates salary

![Viz 5](https://github.com/kautilyak/student-placement-analysis/blob/main/viz/viz5.png)

# Machine Learning

- We have created a model using the random forest regressor (rfr) as it has proved to be the most accurate one among the others. We used another python module saperate from the streamlit app. We did this only because we just need to train the model once before we use it in production. After creating the saved pkl file, we went ahead and used it in our demo:

![Viz 6](https://github.com/kautilyak/student-placement-analysis/blob/main/viz/viz6.png)

- Here, we can tweak the values of the factors affecting the salary and see the predicted salary. 
- According to the model, the most important factors affecting the salary are shown in the following plot

![feature importance](https://github.com/kautilyak/student-placement-analysis/blob/main/viz/fi.png)

# Steps to run the code

- Please install pycaret version 2.3.6, altair, pandas, numpy version 1.19.2 and numba version 0.53.0
- To run the project, navigate to the project folder using command line and use the following command: streamlit run Project-Module.py
