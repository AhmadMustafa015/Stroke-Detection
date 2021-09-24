import csv

csvpath = 'C:/Users/RadioscientificOne/PycharmProjects/Stroke-Detection/Final_output_Prediction/classification_results.csv'
counter0 = 0
counter1 = 0

with open(csvpath) as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] == '0':
            counter0 = counter0 +1
        if row[1] == '1':
            counter1 = counter1 +1
        a =1

print('No inme :',counter0)
print('Yes inme :',counter1)